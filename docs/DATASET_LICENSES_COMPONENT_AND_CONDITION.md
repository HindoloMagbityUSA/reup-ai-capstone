# Component and visible-condition dataset attribution

## HBD — component recognition

- Dataset: Hong Kong University Building Image Dataset (HBD)
- Authors: Mun On Wong, Huaquan Ying, Mengtian Yin, Xiaoyue Yi, Lizhao Xiao, Weilun Duan, Chenchen He, and Llewellyn Tang
- Source: https://github.com/EnochYing/Image2BIM/tree/main/HBD
- Paper: https://doi.org/10.1016/j.autcon.2024.105558
- Source repository license: MIT
- Scope used by ReUP: 11 visible building-component classes

## dacl1k — visible-condition recognition

- Dataset: dacl1k: Real-world bridge damage dataset putting open-source data to the test
- Authors: Johannes Flotzinger, Philipp J. Rösch, Norbert Oswald, and Thomas Braml
- Source: https://github.com/phiyodr/building-inspection-toolkit
- Paper: https://doi.org/10.1016/j.engappai.2024.109106
- Scope used by ReUP: No Damage, Crack, Efflorescence, Spalling, Bars Exposed, and Rust
- Important rights note: the source repository is MIT-licensed, but its `datasets.json` entry for dacl1k still states `license: TODO`. The dataset therefore does not have the same explicit data-license clarity as HBD or the original material dataset.

The original datasets are not redistributed inside the ReUP demo ZIP. dacl1k is used here only for the non-commercial academic capstone and is downloaded directly from the authors' distribution mechanism. Its raw images are not repackaged or redistributed.
