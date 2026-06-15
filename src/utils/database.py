"""
数据库管理模块 - 处理SQLite数据库操作
"""
import sqlite3
import os
from datetime import datetime


class Database:
    def __init__(self, db_path='substation_detection.db'):
        """初始化数据库连接"""
        self.db_path = db_path
        self.conn = None
        self.cursor = None
        self.connect()
        self.create_tables()

    def connect(self):
        """连接到SQLite数据库"""
        try:
            self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self.cursor = self.conn.cursor()
            return True
        except Exception as e:
            print(f"数据库连接失败: {e}")
            return False

    def create_tables(self):
        """创建数据库表"""
        # 用户表
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL,
                role TEXT NOT NULL DEFAULT 'user',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # 检测记录表
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                detection_type TEXT NOT NULL,
                source TEXT,
                object_count INTEGER DEFAULT 0,
                max_confidence REAL DEFAULT 0.0,
                avg_confidence REAL DEFAULT 0.0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')

        self.conn.commit()

    def close(self):
        """关闭数据库连接"""
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()

    # ========== 用户相关操作 ==========

    def add_user(self, username, password, role='user'):
        """添加用户"""
        try:
            self.cursor.execute(
                'INSERT INTO users (username, password, role) VALUES (?, ?, ?)',
                (username, password, role)
            )
            self.conn.commit()
            return self.cursor.lastrowid
        except sqlite3.IntegrityError:
            return None  # 用户名已存在

    def get_user(self, username):
        """获取用户信息"""
        self.cursor.execute(
            'SELECT id, username, password, role FROM users WHERE username = ?',
            (username,)
        )
        return self.cursor.fetchone()

    def verify_user(self, username, password):
        """验证用户登录"""
        user = self.get_user(username)
        if user and user[2] == password:  # user[2] 是密码（已哈希）
            return {
                'id': user[0],
                'username': user[1],
                'role': user[3]
            }
        return None

    def get_all_users(self):
        """获取所有用户"""
        self.cursor.execute('SELECT id, username, role, created_at FROM users')
        return self.cursor.fetchall()

    def delete_user(self, user_id):
        """删除用户"""
        self.cursor.execute('DELETE FROM users WHERE id = ?', (user_id,))
        self.conn.commit()
        return self.cursor.rowcount > 0

    # ========== 检测记录相关操作 ==========

    def add_record(self, user_id, detection_type, source, object_count, max_conf, avg_conf):
        """添加检测记录"""
        try:
            self.cursor.execute(
                '''INSERT INTO records
                (user_id, detection_type, source, object_count, max_confidence, avg_confidence)
                VALUES (?, ?, ?, ?, ?, ?)''',
                (user_id, detection_type, source, object_count, max_conf, avg_conf)
            )
            self.conn.commit()
            return self.cursor.lastrowid
        except Exception as e:
            print(f"添加记录失败: {e}")
            return None

    def get_records(self, user_id=None, limit=None):
        """获取检测记录"""
        query = '''
            SELECT r.id, u.username, r.detection_type, r.source,
                   r.object_count, r.max_confidence, r.avg_confidence, r.created_at
            FROM records r
            JOIN users u ON r.user_id = u.id
        '''
        params = []

        if user_id:
            query += ' WHERE r.user_id = ?'
            params.append(user_id)

        query += ' ORDER BY r.created_at DESC'

        if limit:
            query += ' LIMIT ?'
            params.append(limit)

        self.cursor.execute(query, params)
        return self.cursor.fetchall()

    def get_statistics(self, user_id=None):
        """获取统计数据"""
        query = '''
            SELECT
                COUNT(*) as total_detections,
                SUM(object_count) as total_objects,
                AVG(avg_confidence) as avg_confidence,
                MAX(max_confidence) as max_confidence,
                COUNT(DISTINCT user_id) as user_count
            FROM records
        '''
        params = []

        if user_id:
            query += ' WHERE user_id = ?'
            params.append(user_id)

        self.cursor.execute(query, params)
        result = self.cursor.fetchone()

        return {
            'total_detections': result[0] or 0,
            'total_objects': result[1] or 0,
            'avg_confidence': round(result[2], 4) if result[2] else 0,
            'max_confidence': round(result[3], 4) if result[3] else 0,
            'user_count': result[4] or 0
        }

    def clear_records(self):
        """清空检测记录"""
        self.cursor.execute('DELETE FROM records')
        self.conn.commit()
