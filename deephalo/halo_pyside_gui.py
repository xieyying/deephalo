import os
import sys
     
from PySide6.QtWidgets import (
    QApplication, QWidget, QTabWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton, QTextEdit, QCheckBox, QFileDialog, QGroupBox
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
current_dir = os.path.dirname(os.path.abspath(__file__))
from deephalo.main import pipeline_ms_analyze, pipeline_dereplication, resource_path
from deephalo.parameters import RunParameters

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 只显示严重错误
os.environ['AUTOGRAPH_VERBOSITY'] = '0'   # 禁用 AutoGraph 详细输出


class DeepHaloGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('DeepHalo V1.0.1')
        # 设置窗口小图标
        logo_path = str(resource_path("logo_square.png"))
        if os.path.exists(logo_path):
            from PySide6.QtGui import QIcon
            self.setWindowIcon(QIcon(logo_path))
        self.resize(800, 600)
        self.init_ui()
        # 自动设置 OPENMS_DATA_PATH：优先使用用户环境变量；
        # 如果被 PyInstaller 打包为 --onefile，检查 sys._MEIPASS/OpenMS；
        # 否则尝试包内资源（resource_path）。
        openms_pkg_path = str(resource_path("OpenMS"))
        current_openms = os.environ.get("OPENMS_DATA_PATH")
        if current_openms:
            # 已由用户/系统设置，保留并不覆盖
            print(f"OPENMS_DATA_PATH is already set to: {current_openms}")
        else:
            # 如果运行在 PyInstaller --onefile 的临时目录中，_MEIPASS 会指向解包目录
            meipass = getattr(sys, '_MEIPASS', None)
            candidate = None
            if meipass:
                candidate = os.path.join(meipass, 'OpenMS')
                if os.path.exists(candidate):
                    os.environ['OPENMS_DATA_PATH'] = candidate
                    print(f"Set OPENMS_DATA_PATH to PyInstaller extracted OpenMS: {candidate}")
            # 如果没有通过 PyInstaller 解包到 _MEIPASS，尝试包内资源路径
            if not os.environ.get('OPENMS_DATA_PATH'):
                if os.path.exists(openms_pkg_path):
                    os.environ['OPENMS_DATA_PATH'] = openms_pkg_path
                    print(f"Set OPENMS_DATA_PATH to package resource: {openms_pkg_path}")
                else:
                    # 无法找到 OpenMS 数据，打印警告（不覆盖用户环境变量）
                    print("Warning: OPENMS_DATA_PATH environment variable not found and no share directory was installed. Some functionality might not work as expected.")
                    print("Warning: package OpenMS resource not found; please set OPENMS_DATA_PATH to your OpenMS share directory (e.g. 'C:\\Program Files\\OpenMS\\share\\OpenMS').")

    def init_ui(self):
        layout = QVBoxLayout(self)
        # 加载logo
        logo_label = QLabel()
        logo_path = str(resource_path("logo.png"))
        if os.path.exists(logo_path):
            pixmap = QPixmap(logo_path)
            max_width = 400
            max_height = 100
            scaled_pixmap = pixmap.scaled(max_width, max_height, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            logo_label.setPixmap(scaled_pixmap)
            logo_label.setAlignment(Qt.AlignHCenter)
            layout.addWidget(logo_label, alignment=Qt.AlignHCenter)
        # 标题
        title = QLabel('A deep learning-integrated workflow for high-throughput discovery of halogenated metabolites from HRMS data')
        title.setStyleSheet('font-size: 13px; color: #383C92; margin-top: 0px; margin-bottom: 16px;')
        layout.addWidget(title, alignment=Qt.AlignHCenter)

        subtitle = QLabel('(Note: HRMS data and mzML format needed)')
        subtitle.setStyleSheet('font-size: 12px; color: #57AF67; font-weight: bold; margin-bottom: 16px;')
        layout.addWidget(subtitle, alignment=Qt.AlignHCenter)

        self.tabs = QTabWidget()
        self.detect_tab = QWidget()
        self.derep_tab = QWidget()
        self.tabs.addTab(self.detect_tab, 'Detect')
        self.tabs.addTab(self.derep_tab, 'Dereplicate')
        layout.addWidget(self.tabs)

        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setMinimumHeight(180)
        layout.addWidget(self.log_box)

        self.init_detect_tab()
        self.init_derep_tab()

    def init_detect_tab(self):
        layout = QVBoxLayout()
        # Input file/folder
        h1 = QHBoxLayout()
        h1.addWidget(QLabel('Input (.mzML file or folder):'))
        self.in_path = QLineEdit()
        h1.addWidget(self.in_path)
        btn_file = QPushButton('File')
        btn_file.clicked.connect(self.browse_in_file)
        h1.addWidget(btn_file)
        btn_folder = QPushButton('Folder')
        btn_folder.clicked.connect(self.browse_in_folder)
        h1.addWidget(btn_folder)
        layout.addLayout(h1)
        # Output dir
        h2 = QHBoxLayout()
        h2.addWidget(QLabel('Output Directory:'))
        self.out_path = QLineEdit()
        h2.addWidget(self.out_path)
        btn_out = QPushButton('Browse')
        btn_out.clicked.connect(self.browse_out_folder)
        h2.addWidget(btn_out)
        layout.addLayout(h2)
        # Config file
        h3 = QHBoxLayout()
        h3.addWidget(QLabel('Config File (optional):'))
        self.config_path = QLineEdit()
        h3.addWidget(self.config_path)
        btn_config = QPushButton('Browse')
        btn_config.clicked.connect(self.browse_config_file)
        h3.addWidget(btn_config)
        layout.addLayout(h3)
        # Blank folder
        h4 = QHBoxLayout()
        h4.addWidget(QLabel('Blank Folder (optional):'))
        self.blank_path = QLineEdit()
        h4.addWidget(self.blank_path)
        btn_blank = QPushButton('Browse')
        btn_blank.clicked.connect(self.browse_blank_folder)
        h4.addWidget(btn_blank)
        layout.addLayout(h4)
        # Checkboxes
        self.overwrite_box = QCheckBox('Overwrite Blank')
        self.ms2_box = QCheckBox('Enable MS2 Extraction (only DDA data supported)')
        layout.addWidget(self.overwrite_box)
        layout.addWidget(self.ms2_box)
        # Run button
        btn_run = QPushButton('Run Detect')
        btn_run.clicked.connect(self.run_detect)
        layout.addWidget(btn_run)
        self.detect_tab.setLayout(layout)

    def init_derep_tab(self):
        layout = QVBoxLayout()
        # Project dir
        h1 = QHBoxLayout()
        h1.addWidget(QLabel('Project Directory:'))
        self.derep_out = QLineEdit()
        h1.addWidget(self.derep_out)
        btn_proj = QPushButton('Browse')
        btn_proj.clicked.connect(self.browse_derep_out)
        h1.addWidget(btn_proj)
        layout.addLayout(h1)
        # Config file
        h2 = QHBoxLayout()
        h2.addWidget(QLabel('Config File (optional):'))
        self.derep_config = QLineEdit()
        h2.addWidget(self.derep_config)
        btn_config = QPushButton('Browse')
        btn_config.clicked.connect(self.browse_derep_config)
        h2.addWidget(btn_config)
        layout.addLayout(h2)
        # GNPS folder
        h3 = QHBoxLayout()
        h3.addWidget(QLabel('GNPS Folder (optional):'))
        self.gnps_path = QLineEdit()
        h3.addWidget(self.gnps_path)
        btn_gnps = QPushButton('Browse')
        btn_gnps.clicked.connect(self.browse_gnps_folder)
        h3.addWidget(btn_gnps)
        layout.addLayout(h3)
        # User DB
        h4 = QHBoxLayout()
        h4.addWidget(QLabel('User Database (optional):'))
        self.user_db = QLineEdit()
        h4.addWidget(self.user_db)
        btn_db = QPushButton('Browse')
        btn_db.clicked.connect(self.browse_user_db)
        h4.addWidget(btn_db)
        layout.addLayout(h4)
        # Run button
        btn_run = QPushButton('Run Dereplicate')
        btn_run.clicked.connect(self.run_derep)
        layout.addWidget(btn_run)
        self.derep_tab.setLayout(layout)

    def browse_in_file(self):
        file, _ = QFileDialog.getOpenFileName(self, 'Select mzML File', '', 'mzML Files (*.mzML)')
        if file:
            self.in_path.setText(file)
    def browse_in_folder(self):
        folder = QFileDialog.getExistingDirectory(self, 'Select Folder')
        if folder:
            self.in_path.setText(folder)
    def browse_out_folder(self):
        folder = QFileDialog.getExistingDirectory(self, 'Select Output Directory')
        if folder:
            self.out_path.setText(folder)
    def browse_config_file(self):
        file, _ = QFileDialog.getOpenFileName(self, 'Select Config File', '', 'TOML Files (*.toml);;All Files (*)')
        if file:
            self.config_path.setText(file)
    def browse_blank_folder(self):
        folder = QFileDialog.getExistingDirectory(self, 'Select Blank Folder')
        if folder:
            self.blank_path.setText(folder)
    def browse_derep_out(self):
        folder = QFileDialog.getExistingDirectory(self, 'Select Project Directory')
        if folder:
            self.derep_out.setText(folder)
    def browse_derep_config(self):
        file, _ = QFileDialog.getOpenFileName(self, 'Select Config File', '', 'TOML Files (*.toml);;All Files (*)')
        if file:
            self.derep_config.setText(file)
    def browse_gnps_folder(self):
        folder = QFileDialog.getExistingDirectory(self, 'Select GNPS Folder')
        if folder:
            self.gnps_path.setText(folder)
    def browse_user_db(self):
        file, _ = QFileDialog.getOpenFileName(self, 'Select User Database', '', 'CSV/JSON (*.csv *.json);;All Files (*)')
        if file:
            self.user_db.setText(file)

    def log(self, msg):
        self.log_box.append(msg)

    def run_detect(self):
        try:
            os.makedirs(str(self.out_path.text()), exist_ok=True)
            os.chdir(str(self.out_path.text()))
            user_config = str(self.config_path.text())
            if not user_config:
                user_config = str(resource_path("config.toml"))
            para = RunParameters(user_config=user_config)
            para.args_input = str(self.in_path.text())
            para.args_blank = str(self.blank_path.text()) or None
            para.args_overwrite_blank = self.overwrite_box.isChecked()
            para.args_ms2 = self.ms2_box.isChecked()
            self.log('Running detect...')
            pipeline_ms_analyze(para)
            self.log('Detect completed.')
            self.derep_out.setText(self.out_path.text())
        except Exception as e:
            self.log(f'Error: {e}')

    def run_derep(self):
        try:
            if not self.gnps_path.text() and not self.user_db.text():
                self.log('Error: GNPS folder or user database must be provided.')
                return
            user_config = self.derep_config.text()
            if not user_config:
                user_config = str(resource_path("config.toml"))
            para = RunParameters(user_config=user_config)
            para.args_project_path = self.derep_out.text()
            para.args_GNPS_folder = self.gnps_path.text() or None
            para.args_user_database = self.user_db.text() or None
            self.log('Running dereplicate...')
            pipeline_dereplication(para)
            self.log('Dereplicate completed.')
        except Exception as e:
            self.log(f'Error: {e}')

if __name__ == '__main__':
    app = QApplication(sys.argv)
    gui = DeepHaloGUI()
    gui.show()
    sys.exit(app.exec())
