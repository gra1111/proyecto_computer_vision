<a href="https://x.com/nearcyan/status/1706914605262684394">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/lab-project-dark.png">
    <source media="(prefers-color-scheme: light)" srcset="assets/lab-project-light.png">
    <img alt="Lab Session Image" src="assets/lab-project-light.png">
  </picture>
</a>

**Welcome to the Final Lab Project of Computer *Vision I* at Comillas ICAI**.  
This repository contains the complete implementation of a **classical computer vision system in real time**, developed as part of the final project of the course 📷💻.

The project integrates **camera calibration**, **pattern-based visual security**, and a **tracking system using a Kalman filter**, without relying on Deep Learning techniques.

---

## 📁 Resources

This laboratory project contains the following elements:

- 📄 **Guide**: A `PDF` file with the official project description and requirements.
- 💻 **Scripts**: Python scripts implementing the full system.
- 🎞️ **Data**: Calibration images and stored calibration parameters.
- 🖼️ **Assets**: Images used for documentation and repository styling.
- 📝 **Template**: LaTeX template used to generate the project guide (can be reused for the report).
- 📖 **README**: This file, describing the project structure and functionality.

---

## 🗂️ Project structure

The repository is organized as follows:

```bash
.
├── assets
│   ├── lab-project-dark.png
│   └── lab-project-light.png
│
├── data
│   ├── calibration_00.jpg
│   ├── calibration_01.jpg
│   ├── ...
│   ├── calibration_09.jpg
│   └── camera_calibration_params.npz
│
├── imagenes_con_marca
│   ├── calibration_00_marked.jpg
│   ├── ...
│   └── calibration_09_marked.jpg
│
├── src
│   ├── camera_calibration.py
│   ├── script_principal.py
│   └── test.py
│
├── template
│   └── (LaTeX template for the report)
│
├── Lab_Project.pdf
└── README.md
