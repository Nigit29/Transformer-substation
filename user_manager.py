"""
用户管理模块 - 处理用户登录、注册和认证
"""
import hashlib
import json
from database import Database


class UserManager:
    def __init__(self, db_path='substation_detection.db'):
        """初始化用户管理器"""
        self.db = Database(db_path)
        self.current_user = None

        # 如果没有用户，创建默认管理员账户
        self._create_default_admin()

    def _hash_password(self, password):
        """密码哈希（使用SHA256）"""
        return hashlib.sha256(password.encode('utf-8')).hexdigest()

    def _create_default_admin(self):
        """创建默认管理员账户"""
        users = self.db.get_all_users()
        if not users:
            admin_password = self._hash_password('admin123')
            self.db.add_user('admin', admin_password, 'admin')
            print("默认管理员账户已创建 - 用户名: admin, 密码: admin123")

    def register(self, username, password, role='user'):
        """
        注册新用户

        Args:
            username: 用户名
            password: 明文密码
            role: 角色 ('user' 或 'admin')

        Returns:
            dict: {'success': bool, 'message': str}
        """
        if len(username) < 3:
            return {'success': False, 'message': '用户名至少3个字符'}

        if len(password) < 6:
            return {'success': False, 'message': '密码至少6个字符'}

        if role not in ['user', 'admin']:
            role = 'user'

        hashed_password = self._hash_password(password)
        user_id = self.db.add_user(username, hashed_password, role)

        if user_id:
            return {'success': True, 'message': '注册成功'}
        else:
            return {'success': False, 'message': '用户名已存在'}

    def login(self, username, password):
        """
        用户登录

        Args:
            username: 用户名
            password: 明文密码

        Returns:
            dict: {'success': bool, 'user': dict or None, 'message': str}
        """
        hashed_password = self._hash_password(password)
        user = self.db.verify_user(username, hashed_password)

        if user:
            self.current_user = user
            return {
                'success': True,
                'user': user,
                'message': f'欢迎, {username}!'
            }
        else:
            return {
                'success': False,
                'user': None,
                'message': '用户名或密码错误'
            }

    def logout(self):
        """用户登出"""
        self.current_user = None

    def is_logged_in(self):
        """检查是否已登录"""
        return self.current_user is not None

    def get_current_user(self):
        """获取当前用户信息"""
        return self.current_user

    def is_admin(self):
        """检查当前用户是否是管理员"""
        if self.current_user:
            return self.current_user['role'] == 'admin'
        return False

    def change_password(self, username, old_password, new_password):
        """
        修改密码

        Args:
            username: 用户名
            old_password: 旧密码（明文）
            new_password: 新密码（明文）

        Returns:
            dict: {'success': bool, 'message': str}
        """
        # 验证旧密码
        old_hashed = self._hash_password(old_password)
        user = self.db.get_user(username)

        if not user:
            return {'success': False, 'message': '用户不存在'}

        if user[2] != old_hashed:
            return {'success': False, 'message': '旧密码错误'}

        if len(new_password) < 6:
            return {'success': False, 'message': '新密码至少6个字符'}

        # 更新密码
        new_hashed = self._hash_password(new_password)
        self.db.cursor.execute(
            'UPDATE users SET password = ? WHERE username = ?',
            (new_hashed, username)
        )
        self.db.conn.commit()

        return {'success': True, 'message': '密码修改成功'}

    def get_all_users(self):
        """获取所有用户列表（仅管理员可用）"""
        if not self.is_admin():
            return []

        users = self.db.get_all_users()
        return [{
            'id': u[0],
            'username': u[1],
            'role': u[2],
            'created_at': u[3]
        } for u in users]

    def delete_user(self, user_id):
        """删除用户（仅管理员可用）"""
        if not self.is_admin():
            return {'success': False, 'message': '权限不足'}

        # 不能删除自己
        if user_id == self.current_user['id']:
            return {'success': False, 'message': '不能删除当前登录用户'}

        success = self.db.delete_user(user_id)
        if success:
            return {'success': True, 'message': '用户删除成功'}
        else:
            return {'success': False, 'message': '用户不存在'}
