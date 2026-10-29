T4
| Model | Orig. | Sham | Rem. | Decoy | Only | +15 | +60 | Deletion AUC | Insertion AUC |
|------------------|------|------|------|------|------|------|------|-----------|-----------|
| ResNet50 (orig.) | 0.932 | 0.929 | 0.848 | 0.831 | 0.457 | 0.535 | 0.612 | 0.561 | 0.837 |
| ResNet50 | 0.952 | 0.949 | 0.833 | 0.808 | 0.280 | 0.351 | 0.862 | 0.531 [0.608] | 0.818 [0.607] |
| ResNet50 + SSPANet | 0.954 | 0.952 | 0.840 | 0.818 | 0.271 | 0.346 | 0.856 | 0.508 | 0.825 |
| Y-Net-style (no TFP) | 0.962 | 0.958 | 0.840 | 0.809 | 0.300 | 0.351 | 0.878 | 0.529 | 0.815 |
| **TG-SSPANet** | 0.966 | 0.962 | 0.813 | 0.777 | 0.304 | 0.381 | 0.907 | 0.504 [0.584] | 0.818 [0.584] |
| Oracle GT gate (positive control) | 0.965 | 0.965 | 0.812 | 0.786 | 0.247 | 0.313 | 0.899 | – | – |
| ROI crop (positive control) | 0.956 | 0.956 | 0.648 | 0.644 | 0.335 | 0.675 | 0.954 | – | – |
| VGG16 | 0.944 | 0.942 | 0.820 | 0.805 | 0.286 | 0.371 | 0.863 | 0.444 | 0.755 |
| **VGG16 TG-SSPANet** | 0.964 | 0.961 | 0.801 | 0.789 | 0.301 | 0.412 | 0.937 | 0.442 [0.523] | 0.776 [0.523] |

S2
| Model | Orig. | Sham | Rem. | Decoy | 0 px | 6 px | 15 px | 30 px | 60 px |
|---|---|---|---|---|---|---|---|---|---|
| ResNet50 (orig.) | 0.932 ± 0.018 | 0.929 ± 0.021 | 0.848 ± 0.040 | 0.831 ± 0.041 | 0.457 ± 0.062 | 0.472 ± 0.034 | 0.535 ± 0.057 | 0.575 ± 0.088 | 0.612 ± 0.100 |
| ResNet50 + SSPANet (orig.) | 0.919 ± 0.023 | 0.918 ± 0.026 | 0.861 ± 0.035 | 0.840 ± 0.037 | 0.378 ± 0.071 | 0.416 ± 0.072 | 0.457 ± 0.106 | 0.564 ± 0.072 | 0.710 ± 0.083 |
| ResNet50 | 0.952 ± 0.019 | 0.949 ± 0.014 | 0.833 ± 0.044 | 0.808 ± 0.055 | 0.280 ± 0.047 | 0.304 ± 0.025 | 0.351 ± 0.026 | 0.561 ± 0.058 | 0.862 ± 0.030 |
| ResNet50 + SSPANet | 0.954 ± 0.014 | 0.952 ± 0.012 | 0.840 ± 0.048 | 0.818 ± 0.051 | 0.271 ± 0.050 | 0.286 ± 0.034 | 0.346 ± 0.027 | 0.559 ± 0.040 | 0.856 ± 0.029 |
| Y-Net-style (no TFP) | 0.962 ± 0.016 | 0.958 ± 0.023 | 0.840 ± 0.061 | 0.809 ± 0.050 | 0.300 ± 0.096 | 0.323 ± 0.073 | 0.351 ± 0.049 | 0.553 ± 0.068 | 0.878 ± 0.043 |
| TG-SSPANet | 0.966 ± 0.011 | 0.962 ± 0.009 | 0.813 ± 0.029 | 0.777 ± 0.032 | 0.304 ± 0.064 | 0.340 ± 0.067 | 0.381 ± 0.059 | 0.583 ± 0.046 | 0.907 ± 0.028 |
| Oracle GT gate (positive control) | 0.965 ± 0.011 | 0.965 ± 0.012 | 0.812 ± 0.023 | 0.786 ± 0.034 | 0.247 ± 0.031 | 0.268 ± 0.024 | 0.313 ± 0.043 | 0.521 ± 0.068 | 0.899 ± 0.024 |
| ROI crop, predicted-mask box (positive control) | 0.956 ± 0.005 | 0.956 ± 0.005 | 0.648 ± 0.054 | 0.644 ± 0.056 | 0.335 ± 0.056 | 0.441 ± 0.084 | 0.675 ± 0.093 | 0.935 ± 0.021 | 0.954 ± 0.010 |
| ROI crop, GT-mask box (positive control) | 0.962 ± 0.007 | 0.962 ± 0.008 | 0.642 ± 0.054 | 0.640 ± 0.057 | 0.324 ± 0.056 | 0.429 ± 0.090 | 0.688 ± 0.104 | 0.959 ± 0.008 | 0.962 ± 0.007 |
| VGG16 (orig.) | 0.901 ± 0.026 | 0.898 ± 0.023 | 0.836 ± 0.049 | 0.824 ± 0.052 | 0.289 ± 0.053 | 0.309 ± 0.051 | 0.379 ± 0.069 | 0.544 ± 0.064 | 0.636 ± 0.043 |
| VGG16 | 0.944 ± 0.018 | 0.942 ± 0.019 | 0.820 ± 0.049 | 0.805 ± 0.046 | 0.286 ± 0.037 | 0.293 ± 0.048 | 0.371 ± 0.044 | 0.603 ± 0.064 | 0.863 ± 0.110 |
| VGG16 TG-SSPANet | 0.964 ± 0.020 | 0.961 ± 0.019 | 0.801 ± 0.052 | 0.789 ± 0.049 | 0.301 ± 0.020 | 0.320 ± 0.037 | 0.412 ± 0.048 | 0.629 ± 0.046 | 0.937 ± 0.034 |

S3
| Model | True class | → meningioma | → glioma | → pituitary |
|---|---|---|---|---|
| ResNet50 (orig.) | meningioma | 317 | 340 | 0 |
| ResNet50 (orig.) | glioma | 225 | 928 | 0 |
| ResNet50 (orig.) | pituitary | 286 | 644 | 0 |
| ResNet50 + SSPANet (orig.) | meningioma | 604 | 53 | 0 |
| ResNet50 + SSPANet (orig.) | glioma | 718 | 433 | 2 |
| ResNet50 + SSPANet (orig.) | pituitary | 574 | 354 | 2 |
| ResNet50 | meningioma | 647 | 0 | 10 |
| ResNet50 | glioma | 1044 | 36 | 73 |
| ResNet50 | pituitary | 832 | 12 | 86 |
| ResNet50 + SSPANet | meningioma | 645 | 0 | 12 |
| ResNet50 + SSPANet | glioma | 1067 | 25 | 61 |
| ResNet50 + SSPANet | pituitary | 839 | 12 | 79 |
| Y-Net-style (no TFP) | meningioma | 605 | 0 | 52 |
| Y-Net-style (no TFP) | glioma | 950 | 58 | 145 |
| Y-Net-style (no TFP) | pituitary | 725 | 33 | 172 |
| TG-SSPANet | meningioma | 642 | 4 | 11 |
| TG-SSPANet | glioma | 953 | 145 | 55 |
| TG-SSPANet | pituitary | 794 | 81 | 55 |
| Oracle GT gate (positive control) | meningioma | 653 | 4 | 0 |
| Oracle GT gate (positive control) | glioma | 1117 | 24 | 12 |
| Oracle GT gate (positive control) | pituitary | 887 | 40 | 3 |
| ROI crop, predicted-mask box (positive control) | meningioma | 652 | 5 | 0 |
| ROI crop, predicted-mask box (positive control) | glioma | 885 | 266 | 2 |
| ROI crop, predicted-mask box (positive control) | pituitary | 864 | 61 | 5 |
| ROI crop, GT-mask box (positive control) | meningioma | 655 | 2 | 0 |
| ROI crop, GT-mask box (positive control) | glioma | 914 | 235 | 4 |
| ROI crop, GT-mask box (positive control) | pituitary | 865 | 63 | 2 |
| VGG16 (orig.) | meningioma | 628 | 29 | 0 |
| VGG16 (orig.) | glioma | 985 | 168 | 0 |
| VGG16 (orig.) | pituitary | 783 | 147 | 0 |
| VGG16 | meningioma | 642 | 15 | 0 |
| VGG16 | glioma | 1007 | 144 | 2 |
| VGG16 | pituitary | 803 | 126 | 1 |
| VGG16 TG-SSPANet | meningioma | 638 | 19 | 0 |
| VGG16 TG-SSPANet | glioma | 965 | 187 | 1 |
| VGG16 TG-SSPANet | pituitary | 779 | 151 | 0 |

S4
| Model | Stage | Slice acc. | ECE | AURC | Acc@90% | Patient acc. |
|---|---|---|---|---|---|---|
| ResNet50 (orig.) | before scaling | 0.931 | 0.031 | 0.0236 | 0.960 | 0.936 |
| ResNet50 (orig.) | after scaling | 0.931 | 0.017 | 0.0234 | 0.962 | 0.931 |
| ResNet50 | before scaling | 0.955 | 0.063 | 0.0282 | 0.976 | 0.961 |
| ResNet50 | after scaling | 0.955 | 0.025 | 0.0363 | 0.971 | 0.966 |
| ResNet50 + SSPANet | before scaling | 0.958 | 0.067 | 0.0151 | 0.982 | 0.966 |
| ResNet50 + SSPANet | after scaling | 0.958 | 0.019 | 0.0225 | 0.976 | 0.966 |
| TG-SSPANet | before scaling | 0.968 | 0.074 | 0.0158 | 0.984 | 0.983 |
| TG-SSPANet | after scaling | 0.968 | 0.019 | 0.0193 | 0.982 | 0.987 |
| VGG16 TG-SSPANet | before scaling | 0.965 | 0.110 | 0.0152 | 0.981 | 0.979 |
| VGG16 TG-SSPANet | after scaling | 0.965 | 0.022 | 0.0200 | 0.980 | 0.979 |