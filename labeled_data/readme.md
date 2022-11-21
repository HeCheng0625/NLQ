## Data Introduction

This folder contains some raw data of resumes

label_resume_data.json: a dictionary {label: [exp_str_1, ..., exp_str_n], ...}

example:

```json
{
    ...
    "劳动保障": [
    "深圳市人力资源和社会保障局办公室副主任",
    "深圳市人力资源和社会保障局工伤保险处处长",
    "深圳市财政局社保处副处长",
    ...
    ],
    ...
}

resume_label_data.json: a dictionary {exp_str: [label_1, ..., label_n], ...}

example:

```json
{
    ...
    "同济大学道路工程专业大学本科学生": [
        "学生",
        "交通运输",
        "技术"
    ],
    ...
}
```

resume_label_ner.json: a dictionary {exp_str: {label: [label_1, ..., label_n], ner: str}, ...}

example:

```json
{
    ...
    "深圳市人力资源和社会保障局工伤保险处处长": {
        "label": [
            "保险",
            "劳动保障"
        ],
        "ner": "深圳市L人力资源和社会保障局O工伤保险处S处长P"
    },
    ...
}
```

## Generate Resume

code: generator.py

config: resume.yaml

```python
rg = ResumeGenerator(resume_id, resume_config)
rg.load_resume()
```
