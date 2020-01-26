from imports import *
from nicer import *


if __name__ == '__main__':

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    nicer = NICER(config.can_checkpoint_path, config.nima_checkpoint_path, device=device)

    video_folder = '/autofs/ceph-stud/fischer/thesis/video/test'
    video_list = [x for x in os.listdir(video_folder) if x.split('.')[-1] in ['mp4', 'avi', 'avg']]        # add further formats here

    # --- 1st step: create all frames for video in video_folder

    for idx, video_file in enumerate(sorted(video_list)):
        print("Processing video {} of {}".format(idx, len(video_list)-1))
        cap = cv2.VideoCapture(os.path.join(video_folder, video_file))

        frame_path = os.path.join(video_folder, video_file.split('.')[0]+'_frames')
        if not os.path.exists(frame_path):
            os.mkdir(frame_path)
        else:
            os.rmdir(frame_path)    # delete old
            os.mkdir(frame_path)

        i = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            cv2.imwrite(os.path.join(frame_path, 'frame' + str(i) + '.jpg'), frame)
            i += 1
        cap.release()
        cv2.destroyAllWindows()

    # --- 2nd step: enhance all video frames with given filter combination

    result_path = os.path.join(frame_path, 'results')
    if not os.path.exists(result_path):
        os.mkdir(result_path)

    filter_list = [24, 44, -1, -1, 16, 62, 54, -8]          # compute this using nicer or set manually here
    nicer.set_filters(filter_list)

    for idx, unedited_frame in enumerate(os.listdir(frame_path)):
        if not unedited_frame.endswith('.jpg'): continue

        image = Image.open(os.path.join(frame_path, unedited_frame))
        print("Enhancing img {} of {}".format(idx, len(os.listdir(frame_path))))
        enhanced = nicer.single_image_pass_can(image, abn=True)

        new_img = Image.fromarray(enhanced)
        new_img.save(os.path.join(result_path, unedited_frame.replace('.jpg','_edited.jpg')))


    # --- 3rd step: read all the enhanced frames back in and create a video from them

    edited_frames = []
    sorted_frames = sorted(os.listdir(result_path), key=lambda x: int(x.split('.')[0].replace('frame','').replace('_edited','')))
    for idx, edited_frame in enumerate(sorted_frames):
        if not edited_frame.endswith('.jpg'): continue

        cv_frame = cv2.imread(os.path.join(result_path, edited_frame))
        height, width, layers = cv_frame.shape
        size = (width, height)
        edited_frames.append(cv_frame)
        print("Reading frame {} of {}".format(idx,len(sorted_frames)))

    print("Writing video")
    frame_rate = 25.0       # TODO: frames currently has to be set manually, to match the source video. In nautilus, get framerate of file via properties->video
    video_destination = os.path.join(video_folder, 'video_edited.avi')
    out = cv2.VideoWriter(video_destination, cv2.VideoWriter_fourcc(*'DIVX'), frame_rate, size)

    for i in range(len(edited_frames)):
        out.write(edited_frames[i])
    out.release()

