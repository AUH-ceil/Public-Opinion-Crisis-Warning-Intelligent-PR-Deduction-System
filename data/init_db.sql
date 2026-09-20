-- ============================================================
-- data/init_db.sql - 数据库初始化SQL脚本
-- 直接在MySQL中执行：mysql -u root -p < init_db.sql
-- ============================================================

-- 建库
CREATE DATABASE IF NOT EXISTS crisis_db DEFAULT CHARSET utf8mb4 COLLATE utf8mb4_unicode_ci;
USE crisis_db;

-- ==================== 表1：媒体权重表 ====================
DROP TABLE IF EXISTS media_weights;
CREATE TABLE media_weights (
    id INT AUTO_INCREMENT PRIMARY KEY,
    platform VARCHAR(32) NOT NULL COMMENT '平台名称：微博/抖音/小红书/新闻媒体/贴吧/知乎',
    account_level VARCHAR(16) DEFAULT '腰部' COMMENT '账号级别：头部/腰部/尾部',
    weight FLOAT DEFAULT 1.0 COMMENT '媒体权重系数（头部>腰部>尾部）',
    avg_reach INT DEFAULT 0 COMMENT '平均触达人数',
    category VARCHAR(64) DEFAULT '综合' COMMENT '领域分类'
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='媒体权重表';

-- ==================== 表2：历史结构化舆情数据表 ====================
DROP TABLE IF EXISTS history_records;
CREATE TABLE history_records (
    id INT AUTO_INCREMENT PRIMARY KEY,
    event_name VARCHAR(128) NOT NULL COMMENT '事件名称',
    category VARCHAR(64) DEFAULT '产品质量' COMMENT '事件分类',
    platform VARCHAR(32) DEFAULT '微博' COMMENT '首发平台',
    negative_score FLOAT DEFAULT 0 COMMENT '负面分值 0-1',
    media_weight FLOAT DEFAULT 1.0 COMMENT '媒体权重',
    risk_index FLOAT DEFAULT 0 COMMENT '综合风险指数 0-100',
    hot_search_hours INT DEFAULT 0 COMMENT '登上热搜耗时（小时）',
    resolution TEXT COMMENT '公关处理方案',
    occurred_at DATE COMMENT '发生日期'
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='历史舆情数据表';

-- ==================== 样例数据：媒体权重 ====================
INSERT INTO media_weights (platform, account_level, weight, avg_reach, category) VALUES
('微博', '头部', 2.5, 5000000, '综合'),
('微博', '腰部', 1.5, 500000, '综合'),
('微博', '尾部', 0.8, 10000, '综合'),
('抖音', '头部', 2.8, 8000000, '综合'),
('抖音', '腰部', 1.6, 800000, '综合'),
('抖音', '尾部', 0.9, 20000, '综合'),
('小红书', '头部', 2.0, 2000000, '美妆'),
('小红书', '腰部', 1.3, 300000, '美妆'),
('新闻媒体', '头部', 3.0, 10000000, '综合'),
('新闻媒体', '腰部', 2.0, 2000000, '综合'),
('知乎', '头部', 1.8, 1500000, '科技'),
('贴吧', '腰部', 1.2, 200000, '综合');

-- ==================== 样例数据：历史舆情案例 ====================
INSERT INTO history_records (event_name, category, platform, negative_score, media_weight, risk_index, hot_search_hours, resolution, occurred_at) VALUES
('某奶茶品牌食品安全事件', '食品安全', '微博', 0.85, 2.8, 78.5, 4, '立即下架问题产品，公开道歉，邀请第三方检测，赔偿消费者，CEO直播致歉', '2024-03-15'),
('某手机电池爆炸事件', '产品质量', '新闻媒体', 0.92, 3.0, 88.2, 2, '全球召回问题批次，成立专项调查组，公布调查结果，升级品控流程，赔偿用户', '2024-01-20'),
('某酒店服务纠纷事件', '服务纠纷', '抖音', 0.72, 2.3, 62.0, 8, '公开道歉信，免去涉事人员职务，全面服务培训，补偿住客，接受社会监督', '2024-05-10'),
('某电商平台售假事件', '产品质量', '微博', 0.88, 2.6, 80.5, 3, '封禁涉事店铺，设立假货举报基金，引入区块链溯源，假一赔十承诺', '2024-02-28'),
('某快餐品牌卫生问题', '食品安全', '抖音', 0.90, 2.9, 85.0, 2, '停业整顿涉事门店，全员卫生培训，安装明厨亮灶监控，邀请消费者参观后厨', '2024-06-01'),
('某汽车品牌刹车失灵', '产品质量', '新闻媒体', 0.95, 3.0, 92.0, 1, '全球召回，联合第三方检测机构调查，CEO公开致歉，更换供应商，加强质检', '2024-04-05'),
('某航空公司超售事件', '服务纠纷', '微博', 0.70, 2.5, 65.0, 3, '调整超售政策，提高补偿标准，员工沟通培训，设立旅客权益保障基金', '2024-07-12'),
('某化妆品过敏事件', '产品质量', '小红书', 0.65, 2.0, 55.0, 12, '下架问题批次，公布成分检测报告，完善过敏提示标签，赔偿医疗费用', '2024-03-28');
