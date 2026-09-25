import cv2
import numpy as np
import onnxruntime as ort


class FaceDetector:

    def __init__(self, model_path="model/Face/det_500m.onnx", conf_threshold=0.5, nms_threshold=0.4):
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold

        # Khởi tạo ONNX Runtime Session
        self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider']) #chạy trên CPU
        
        
        # Lấy thông tin Input
        input_cfg = self.session.get_inputs()[0]
        self.input_name = input_cfg.name
        self.input_shape = input_cfg.shape  # [1, 3, height, width]
        
        # Cấu hình Anchor stride & num_anchors cho SCRFD
        self.fmc = 3
        self._feat_stride_fpn = [8, 16, 32]
        self._num_anchors = 2
        
        # Cấu hình kích thước chuẩn cho SCRFD (thường là 640x640)
        self.target_size = (640, 640)
        
        # TÍNH TRƯỚC (Pre-compute) ANCHOR CENTERS ĐỂ TỐI ƯU TỐC ĐỘ
        self.center_cache = self._generate_anchor_centers(self.target_size)

    def _generate_anchor_centers(self, target_size):
        """Tính trước tọa độ Anchor Centers cho các layer FPN để tránh tính lại mỗi frame"""
        center_cache = {}
        input_height, input_width = target_size

        for stride in self._feat_stride_fpn:
            height = input_height // stride
            width = input_width // stride

            anchor_centers = np.stack(np.mgrid[:height, :width][::-1], axis=-1).astype(np.float32)
            anchor_centers = (anchor_centers * stride).reshape((-1, 2))
            
            if self._num_anchors > 1:
                anchor_centers = np.stack([anchor_centers] * self._num_anchors, axis=1).reshape((-1, 2))

            center_cache[stride] = anchor_centers
        return center_cache

    def _distance2bbox(self, points, distance):
        """Chuyển đổi khoảng cách từ anchor point sang Bounding Box (x1, y1, x2, y2)"""
        x1 = points[:, 0] - distance[:, 0]
        y1 = points[:, 1] - distance[:, 1]
        x2 = points[:, 0] + distance[:, 2]
        y2 = points[:, 1] + distance[:, 3]
        return np.stack([x1, y1, x2, y2], axis=-1)

    def _distance2kps(self, points, distance):
        """Chuyển đổi khoảng cách sang 5 điểm Landmarks trên mặt: shape (N, 5, 2)"""
        preds = []
        for i in range(0, distance.shape[1], 2):
            px = points[:, 0] + distance[:, i]
            py = points[:, 1] + distance[:, i + 1]
            preds.append(px)
            preds.append(py)
        kps = np.stack(preds, axis=-1)  # shape (N, 10)
        return kps.reshape(-1, 5, 2)    # reshape thành (N, 5, 2)

    def detect(self, img):
        """
        Nhận diện khuôn mặt từ ảnh đầu vào
        :param img: Ảnh BGR từ OpenCV (Numpy Array)
        :return: (bboxes, kpss)
                 bboxes: numpy array shape (N, 5) -> [x1, y1, x2, y2, score]
                 kpss: numpy array shape (N, 5, 2) -> [[x1, y1], ..., [x5, y5]]
        """
        h_img, w_img = img.shape[:2]

        # 1. Preprocess: Aspect-ratio Preserving Resize
        target_w, target_h = self.target_size
        im_ratio = float(h_img) / float(w_img)
        model_ratio = float(target_h) / float(target_w)

        if im_ratio > model_ratio:
            new_h = target_h
            new_w = int(new_h / im_ratio)
        else:
            new_w = target_w
            new_h = int(new_w * im_ratio)

        det_scale = float(new_h) / float(h_img)
        resized_img = cv2.resize(img, (new_w, new_h))

        # Pad ảnh để vừa khung 640x640
        det_img = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        det_img[:new_h, :new_w, :] = resized_img

        # Normalize tiêu chuẩn SCRFD (mean 127.5, std 128.0)
        input_blob = cv2.dnn.blobFromImage(
            det_img, 1.0 / 128.0, self.target_size, (127.5, 127.5, 127.5), swapRB=True
        )

        # 2. Forward ONNX Model
        outputs = self.session.run(None, {self.input_name: input_blob})

        # 3. Post-process (Giải mã Output Tensor)
        scores_list = []
        bboxes_list = []
        kpss_list = []

        for idx, stride in enumerate(self._feat_stride_fpn):
            scores = outputs[idx]
            bbox_preds = outputs[idx + self.fmc] * stride
            kps_preds = outputs[idx + self.fmc * 2] * stride

            # Làm phẳng score array để lọc ngưỡng
            scores = scores.flatten()

            pos_inds = np.where(scores >= self.conf_threshold)[0]
            if len(pos_inds) == 0:
                continue

            scores = scores[pos_inds]
            bbox_preds = bbox_preds[pos_inds]
            kps_preds = kps_preds[pos_inds]
            
            # Lấy anchor center đã tính sẵn
            anchor_centers = self.center_cache[stride][pos_inds]

            bboxes = self._distance2bbox(anchor_centers, bbox_preds)
            kpss = self._distance2kps(anchor_centers, kps_preds)

            scores_list.append(scores)
            bboxes_list.append(bboxes)
            kpss_list.append(kpss)

        if not scores_list:
            return np.empty((0, 5)), np.empty((0, 5, 2))

        scores = np.concatenate(scores_list, axis=0)
        bboxes = np.concatenate(bboxes_list, axis=0)
        kpss = np.concatenate(kpss_list, axis=0)

        # Trả BBox và Landmark về tỷ lệ ảnh gốc (Un-scale)
        bboxes /= det_scale
        kpss /= det_scale

        # 4. Non-Maximum Suppression (NMS)
        # FIX LỖI: cv2.dnn.NMSBoxes yêu cầu định dạng [x, y, w, h]
        nms_boxes = bboxes.copy()
        nms_boxes[:, 2] = nms_boxes[:, 2] - nms_boxes[:, 0]  # width = x2 - x1
        nms_boxes[:, 3] = nms_boxes[:, 3] - nms_boxes[:, 1]  # height = y2 - y1

        keep = cv2.dnn.NMSBoxes(
            nms_boxes.tolist(),
            scores.tolist(),
            self.conf_threshold,
            self.nms_threshold
        )

        if len(keep) > 0:
            keep = keep.flatten()
            dets = np.hstack((bboxes[keep], scores[keep, None]))
            return dets, kpss[keep]
        else:
            return np.empty((0, 5)), np.empty((0, 5, 2))