# /root/projects/second.pytorch/second/models/second_carv1/car_onestage4test/voxelnet-30160.tckpt
# /root/projects/second.pytorch/second/configs/view_car.fhd.config
# /dev/shm/dsets/KITTI/kitti_infos_test.pkl

from second.data.kitti_dataset import KittiDataset, kitti_anno_to_label_file
import pickle
root_path = "/dev/shm/dsets/KITTI"
info_path = "/dev/shm/dsets/KITTI/kitti_infos_test.pkl"
det_pkl = "/root/projects/second.pytorch/second/models/second_carv1/car_onestage4test/eval_results/step_30160/result.pkl"
dest_folder = "submit_results"
with open(det_pkl, 'rb') as f:
    det_results = pickle.load(f)
    print(det_results[0].keys())

kd = KittiDataset(root_path, info_path, class_names=["Car"])
annos = kd.convert_detection_to_kitti_annos(det_results)
with open("view_result.pkl", 'wb') as f:
    pickle.dump(annos, f)
kitti_anno_to_label_file(annos, dest_folder)