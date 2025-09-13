# NTUEE DJJ Character Verification Lab 25SPRING

##### author: B10901176 蔡弘祥

### Data Explanation

There are 9 sets of chinese character and each set contains 100 data.

Data in `database/` are the characters written by the professor.

Data in `testbase/` are the characters imitated by the others.

Below is the table explaining how we assign the data.

| directory             | training data file                  | test data file                       |
|-----------------------|-------------------------------------|--------------------------------------|
| `data/[id]/database/` | `base_1_1_[id]` to `base_1_25_[id]` | `base_1_26_[id]` to `base_1_50_[id]` |
| `data/[id]/testcase/` | `base_1_1_[id]` to `base_25_1_[id]` | `base_26_1_[id]` to `base_50_1_[id]` |

### TODO

Use machine learning technique to differentiate which characters are real and which characters are imitated.
