"""
变电站异物检测系统 - 主程序入口
包含登录界面和应用程序启动逻辑
"""
import sys
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QLineEdit, QMessageBox, QCheckBox,
    QTabWidget, QGroupBox, QDialog, QFormLayout
)
from PyQt5.QtGui import QFont, QIcon
from PyQt5.QtCore import Qt

from user_manager import UserManager
from main_window import MainWindow
from config import Config


class LoginWindow(QMainWindow):
    """登录窗口"""

    def __init__(self):
        super().__init__()
        self.user_manager = UserManager(Config.DB_PATH)
        self.current_user = None
        self.init_ui()

    def init_ui(self):
        """初始化登录界面"""
        self.setWindowTitle("变电站异物检测系统 - 登录")
        self.setMinimumWidth(400)
        self.setMinimumHeight(300)

        # 创建中心部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # 主布局
        main_layout = QVBoxLayout(central_widget)

        # 标题
        title_label = QLabel("变电站异物检测系统")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setFont(QFont("Arial", 20, QFont.Bold))
        main_layout.addWidget(title_label)

        # 创建选项卡
        self.tab_widget = QTabWidget()

        # 登录选项卡
        login_tab = self.create_login_tab()
        self.tab_widget.addTab(login_tab, "登录")

        # 注册选项卡
        register_tab = self.create_register_tab()
        self.tab_widget.addTab(register_tab, "注册")

        main_layout.addWidget(self.tab_widget)

        # 版本信息
        version_label = QLabel("版本 1.0.0")
        version_label.setAlignment(Qt.AlignCenter)
        version_label.setStyleSheet("color: #888;")
        main_layout.addWidget(version_label)

    def create_login_tab(self):
        """创建登录选项卡"""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # 登录表单
        form_group = QGroupBox("用户登录")
        form_layout = QFormLayout()

        # 用户名
        self.login_username = QLineEdit()
        self.login_username.setPlaceholderText("请输入用户名")
        form_layout.addRow("用户名:", self.login_username)

        # 密码
        self.login_password = QLineEdit()
        self.login_password.setPlaceholderText("请输入密码")
        self.login_password.setEchoMode(QLineEdit.Password)
        form_layout.addRow("密码:", self.login_password)

        # 记住密码
        self.remember_password = QCheckBox("记住密码")
        form_layout.addRow(self.remember_password)

        form_group.setLayout(form_layout)
        layout.addWidget(form_group)

        # 登录按钮
        login_button = QPushButton("登录")
        login_button.setMinimumHeight(40)
        login_button.setFont(QFont("Arial", 12))
        login_button.clicked.connect(self.login)
        layout.addWidget(login_button)

        layout.addStretch()

        return tab

    def create_register_tab(self):
        """创建注册选项卡"""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # 注册表单
        form_group = QGroupBox("用户注册")
        form_layout = QFormLayout()

        # 用户名
        self.register_username = QLineEdit()
        self.register_username.setPlaceholderText("请输入用户名（至少3个字符）")
        form_layout.addRow("用户名:", self.register_username)

        # 密码
        self.register_password = QLineEdit()
        self.register_password.setPlaceholderText("请输入密码（至少6个字符）")
        self.register_password.setEchoMode(QLineEdit.Password)
        form_layout.addRow("密码:", self.register_password)

        # 确认密码
        self.register_confirm = QLineEdit()
        self.register_confirm.setPlaceholderText("请再次输入密码")
        self.register_confirm.setEchoMode(QLineEdit.Password)
        form_layout.addRow("确认密码:", self.register_confirm)

        form_group.setLayout(form_layout)
        layout.addWidget(form_group)

        # 注册按钮
        register_button = QPushButton("注册")
        register_button.setMinimumHeight(40)
        register_button.setFont(QFont("Arial", 12))
        register_button.clicked.connect(self.register)
        layout.addWidget(register_button)

        layout.addStretch()

        # 提示信息
        info_label = QLabel("默认管理员账户:\n用户名: admin\n密码: admin123")
        info_label.setAlignment(Qt.AlignCenter)
        info_label.setStyleSheet("color: #666; padding: 10px; background-color: #f0f0f0; border-radius: 5px;")
        layout.addWidget(info_label)

        return tab

    def login(self):
        """处理登录"""
        username = self.login_username.text().strip()
        password = self.login_password.text().strip()

        if not username:
            QMessageBox.warning(self, "警告", "请输入用户名")
            return

        if not password:
            QMessageBox.warning(self, "警告", "请输入密码")
            return

        # 验证登录
        result = self.user_manager.login(username, password)

        if result['success']:
            # 登录成功
            if self.remember_password.isChecked():
                self.save_credentials(username, password)

            self.current_user = result['user']
            self.open_main_window()
        else:
            QMessageBox.warning(self, "登录失败", result['message'])

    def register(self):
        """处理注册"""
        username = self.register_username.text().strip()
        password = self.register_password.text().strip()
        confirm = self.register_confirm.text().strip()

        if not username or len(username) < 3:
            QMessageBox.warning(self, "警告", "用户名至少3个字符")
            return

        if not password or len(password) < 6:
            QMessageBox.warning(self, "警告", "密码至少6个字符")
            return

        if password != confirm:
            QMessageBox.warning(self, "警告", "两次输入的密码不一致")
            return

        # 注册用户
        result = self.user_manager.register(username, password, 'user')

        if result['success']:
            QMessageBox.information(self, "注册成功", "注册成功！请使用新账户登录")
            # 切换到登录选项卡
            self.tab_widget.setCurrentIndex(0)
            self.login_username.setText(username)
            self.register_username.clear()
            self.register_password.clear()
            self.register_confirm.clear()
        else:
            QMessageBox.warning(self, "注册失败", result['message'])

    def save_credentials(self, username, password):
        """保存凭据到本地文件（简单实现）"""
        try:
            import os
            import json

            config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.login_config')
            with open(config_path, 'w') as f:
                json.dump({'username': username, 'password': password}, f)
        except Exception as e:
            print(f"保存凭据失败: {e}")

    def load_credentials(self):
        """从本地文件加载凭据"""
        try:
            import os
            import json

            config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.login_config')
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    return json.load(f)
        except Exception as e:
            print(f"加载凭据失败: {e}")
        return None

    def open_main_window(self):
        """打开主窗口"""
        self.main_window = MainWindow(self.current_user)
        self.main_window.show()
        self.close()


def main():
    """主函数"""
    app = QApplication(sys.argv)

    # 设置应用样式
    app.setStyle('Fusion')

    # 创建登录窗口
    login_window = LoginWindow()
    login_window.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
