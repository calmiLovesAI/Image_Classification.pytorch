import torch


class Classify:
    def __init__(self, model, images, class_name, print_on=False):
        self.model = model
        self.model.eval()
        self.images = images
        self.class_name = class_name  # 类别名
        self.print_on = print_on

    def _show_result(self, prediction):
        pred, prob = prediction
        pred = pred.cpu().numpy()
        prob = prob.cpu().numpy()
        for i in range(pred.shape[0]):
            print("第{}张图片的类别是{}，预测概率为：{}".format(i, self.class_name[pred[i]], prob[i]))

    def process_image(self):
        with torch.no_grad():
            pred_logits = self.model(self.images)
            prob = torch.nn.functional.softmax(pred_logits, dim=1)
            pred = torch.argmax(pred_logits, dim=1)
        if self.print_on:
            self._show_result([pred, prob])
        return pred, prob
