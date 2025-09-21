import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def compute_iou(box1, box2):
    """Compute the intersection over union of two set of boxes, each box is [x1,y1,x2,y2].
    Args:
      box1: (tensor) bounding boxes, sized [N,4].
      box2: (tensor) bounding boxes, sized [M,4].
    Return:
      (tensor) iou, sized [N,M].
    """
    N = box1.size(0)
    M = box2.size(0)

    lt = torch.max(
        box1[:, :2].unsqueeze(1).expand(N, M, 2), 
        box2[:, :2].unsqueeze(0).expand(N, M, 2), 
    )

    rb = torch.min(
        box1[:, 2:].unsqueeze(1).expand(N, M, 2),  
        box2[:, 2:].unsqueeze(0).expand(N, M, 2), 
    )

    wh = rb - lt  
    wh[wh < 0] = 0  
    inter = wh[:, :, 0] * wh[:, :, 1]  

    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1]) 
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
    area1 = area1.unsqueeze(1).expand_as(inter)  
    area2 = area2.unsqueeze(0).expand_as(inter) 

    iou = inter / (area1 + area2 - inter)
    return iou


class YoloLoss(nn.Module):

    def __init__(self, S, B, l_coord, l_noobj):
        super(YoloLoss, self).__init__()
        self.S = S
        self.B = B
        self.l_coord = l_coord
        self.l_noobj = l_noobj

    def xywh2xyxy(self, boxes):
    
          x = boxes[:, 0] / self.S
          y = boxes[:, 1] / self.S
          w = boxes[:, 2]
          h = boxes[:, 3]

          x1 = x - 0.5 * w
          y1 = y - 0.5 * h
          x2 = x + 0.5 * w
          y2 = y + 0.5 * h

          boxes_converted = torch.stack([x1, y1, x2, y2], dim=1)
          return boxes_converted


    def find_best_iou_boxes(self, pred_box_list, box_target):
    
          reshaped_preds = []
          
          for i in range(self.B):
              box = pred_box_list[i][..., :4] 
              reshaped_preds.append(self.xywh2xyxy(box.reshape(-1, 4)))  

          target_boxes = self.xywh2xyxy(box_target.reshape(-1, 4)) 

          
          ious = [compute_iou(pred_boxes, target_boxes).diagonal().unsqueeze(1) for pred_boxes in reshaped_preds] 
          ious_tensor = torch.cat(ious, dim=1)  

          best_ious, best_indices = torch.max(ious_tensor, dim=1)  

       
          all_boxes = torch.stack([b.reshape(-1, 5) for b in pred_box_list], dim=1) 
          best_boxes = all_boxes[torch.arange(best_indices.shape[0]), best_indices]  

          return best_ious.unsqueeze(1), best_boxes


    def get_class_prediction_loss(self, classes_pred, classes_target, has_object_map):
    

          mask = has_object_map.unsqueeze(-1)  
          diff = mask * (classes_pred - classes_target)
          loss = torch.sum(diff ** 2)
          return loss


    def get_no_object_loss(self, pred_boxes_list, has_object_map):
          
          loss = 0.0
          noobj_mask = ~has_object_map 

          for b in pred_boxes_list:
              
              pred_conf = b[..., 4]  # shape: (N, S, S)
              
              loss += torch.sum((pred_conf[noobj_mask]) ** 2)

          return loss


    def get_contain_conf_loss(self, box_pred_conf, box_target_conf):
   
          
          box_target_conf = box_target_conf.detach()
          loss = torch.sum((box_pred_conf - box_target_conf) ** 2)
          return loss

      

    def get_regression_loss(self, box_pred_response, box_target_response):
    
          pred_xy = box_pred_response[:, :2]
          pred_wh = torch.sqrt(torch.abs(box_pred_response[:, 2:])) 
          target_xy = box_target_response[:, :2]
          target_wh = torch.sqrt(torch.abs(box_target_response[:, 2:]))

          loss_xy = torch.sum((pred_xy - target_xy) ** 2)
          loss_wh = torch.sum((pred_wh - target_wh) ** 2)

          reg_loss = loss_xy + loss_wh
          return reg_loss



    def forward(self, pred_tensor, target_boxes, target_cls, has_object_map):
        N = pred_tensor.size(0)                   # batch size
    
     
        pred_boxes_list = [pred_tensor[..., b*5:(b+1)*5] for b in range(self.B)]
        pred_cls = pred_tensor[..., self.B*5:]
    
      
        cls_loss   = self.get_class_prediction_loss(pred_cls, target_cls, has_object_map)
        no_obj_loss= self.get_no_object_loss(pred_boxes_list, has_object_map)
    
        mask = has_object_map
        target_boxes_masked = target_boxes[mask]
        if target_boxes_masked.numel() == 0:
      
            return dict(total_loss=cls_loss + self.l_noobj * no_obj_loss,
                        reg_loss=torch.tensor(0.0, device=cls_loss.device),
                        containing_obj_loss=torch.tensor(0.0, device=cls_loss.device),
                        no_obj_loss=no_obj_loss,
                        cls_loss=cls_loss)
    
        pred_boxes_list_masked = [b[mask] for b in pred_boxes_list]
        best_ious, best_boxes  = self.find_best_iou_boxes(pred_boxes_list_masked, target_boxes_masked)
    
        reg_loss        = self.get_regression_loss(best_boxes[:, :4], target_boxes_masked)
        pred_conf       = best_boxes[:, 4].unsqueeze(1)
        contain_obj_loss= self.get_contain_conf_loss(pred_conf, best_ious)
    
       
        K = mask.sum().clamp(min=1).float()     
        cls_loss        = cls_loss        / K
        reg_loss        = reg_loss        / K
        contain_obj_loss= contain_obj_loss/ K
        no_obj_loss     = no_obj_loss     / (N * self.S * self.S)  
    
        
        total_loss = (self.l_coord * reg_loss +
                       contain_obj_loss +
                       self.l_noobj * no_obj_loss +
                       cls_loss)
    
        return dict(total_loss=total_loss,
                    reg_loss=reg_loss,
                    containing_obj_loss=contain_obj_loss,
                    no_obj_loss=no_obj_loss,
                    cls_loss=cls_loss)
