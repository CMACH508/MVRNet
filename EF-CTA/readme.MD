## Data Processing

### Image Processing

- **Without skull CTA**: Obtain Without skull CTA data by running `step1_dicom_to_nii.py`, `step2_StripSkullCT`, and `step3_image_resample.py`.
- **EF-CTA**: Obtain augmented data by running `step1_dicom_to_nii.py`, `step3_image_resample.py`, and `step4_FE_CTA.py`.

### Label Processing

- `step_gt_1resample.py`: Resample the ground truth (GT).
- `step_gt_2denoise_bbox.py`: Denoise the ground truth and obtain lesion bounding boxes (bbox).