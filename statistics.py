"""
统计模块 - 处理检测记录的统计和分析
"""
from database import Database


class Statistics:
    def __init__(self, db_path='substation_detection.db'):
        """
        初始化统计模块

        Args:
            db_path: 数据库文件路径
        """
        self.db = Database(db_path)

    def get_overall_statistics(self, user_id=None):
        """
        获取总体统计信息

        Args:
            user_id: 用户ID（可选，为空则统计所有用户）

        Returns:
            dict: 统计信息
        """
        return self.db.get_statistics(user_id)

    def get_recent_records(self, limit=10, user_id=None):
        """
        获取最近的检测记录

        Args:
            limit: 记录数量限制
            user_id: 用户ID（可选）

        Returns:
            list: 检测记录列表
        """
        return self.db.get_records(user_id, limit)

    def get_detection_type_statistics(self, user_id=None):
        """
        按检测类型统计

        Args:
            user_id: 用户ID（可选）

        Returns:
            dict: 各类型的统计信息
        """
        query = '''
            SELECT
                detection_type,
                COUNT(*) as count,
                SUM(object_count) as total_objects,
                AVG(avg_confidence) as avg_confidence,
                MAX(max_confidence) as max_confidence
            FROM records
        '''
        params = []

        if user_id:
            query += ' WHERE user_id = ?'
            params.append(user_id)

        query += ' GROUP BY detection_type ORDER BY count DESC'

        self.db.cursor.execute(query, params)
        results = self.db.cursor.fetchall()

        return {
            row[0]: {
                'count': row[1],
                'total_objects': row[2] or 0,
                'avg_confidence': round(row[3], 4) if row[3] else 0,
                'max_confidence': round(row[4], 4) if row[4] else 0
            }
            for row in results
        }

    def get_time_statistics(self, days=7, user_id=None):
        """
        获取按时间统计的信息

        Args:
            days: 统计最近多少天
            user_id: 用户ID（可选）

        Returns:
            dict: 按日期分组的统计信息
        """
        query = '''
            SELECT
                DATE(created_at) as date,
                COUNT(*) as count,
                SUM(object_count) as total_objects
            FROM records
        '''
        params = []

        conditions = []
        if user_id:
            conditions.append('user_id = ?')
            params.append(user_id)

        if days:
            conditions.append(f"created_at >= date('now', '-{days} days')")

        if conditions:
            query += ' WHERE ' + ' AND '.join(conditions)

        query += ' GROUP BY DATE(created_at) ORDER BY date DESC'

        self.db.cursor.execute(query, params)
        results = self.db.cursor.fetchall()

        return {
            row[0]: {
                'count': row[1],
                'total_objects': row[2] or 0
            }
            for row in results
        }

    def get_user_statistics(self, user_id=None):
        """
        获取用户统计信息

        Args:
            user_id: 用户ID（可选）

        Returns:
            list: 用户统计列表
        """
        query = '''
            SELECT
                u.username,
                u.role,
                COUNT(r.id) as detection_count,
                SUM(r.object_count) as total_objects,
                AVG(r.avg_confidence) as avg_confidence
            FROM users u
            LEFT JOIN records r ON u.id = r.user_id
        '''

        params = []
        if user_id:
            query += ' WHERE u.id = ?'
            params.append(user_id)

        query += ' GROUP BY u.id ORDER BY detection_count DESC'

        self.db.cursor.execute(query, params)
        results = self.db.cursor.fetchall()

        return [{
            'username': row[0],
            'role': row[1],
            'detection_count': row[2] or 0,
            'total_objects': row[3] or 0,
            'avg_confidence': round(row[4], 4) if row[4] else 0
        } for row in results]

    def get_confidence_distribution(self, user_id=None):
        """
        获取置信度分布统计

        Args:
            user_id: 用户ID（可选）

        Returns:
            dict: 各置信度区间的统计
        """
        query = '''
            SELECT
                CASE
                    WHEN max_confidence >= 0.9 THEN '90-100%'
                    WHEN max_confidence >= 0.7 THEN '70-89%'
                    WHEN max_confidence >= 0.5 THEN '50-69%'
                    ELSE '<50%'
                END as range,
                COUNT(*) as count
            FROM records
        '''
        params = []

        if user_id:
            query += ' WHERE user_id = ?'
            params.append(user_id)

        query += ' GROUP BY range ORDER BY MAX(max_confidence) DESC'

        self.db.cursor.execute(query, params)
        results = self.db.cursor.fetchall()

        return {row[0]: row[1] for row in results}

    def format_statistics_text(self, stats):
        """
        格式化统计信息为文本

        Args:
            stats: 统计信息字典

        Returns:
            str: 格式化的文本
        """
        lines = [
            "=== 检测统计 ===",
            f"检测次数: {stats.get('total_detections', 0)}",
            f"异物总数: {stats.get('total_objects', 0)}",
            f"平均置信度: {stats.get('avg_confidence', 0):.2%}",
            f"最高置信度: {stats.get('max_confidence', 0):.2%}",
            f"使用人数: {stats.get('user_count', 0)}"
        ]
        return '\n'.join(lines)

    def clear_all_records(self):
        """清空所有检测记录"""
        self.db.clear_records()
