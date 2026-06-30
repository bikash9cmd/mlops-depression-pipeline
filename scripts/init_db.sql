-- ============================================================
-- MLOps Depression Prediction – MariaDB Initialisation
-- Star Schema + Application Tables
-- BCU CMP5366 – Bikash Kushwaha
-- ============================================================

-- Create Airflow database (if not exists)
CREATE DATABASE IF NOT EXISTS airflow CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
CREATE USER IF NOT EXISTS 'airflow'@'%' IDENTIFIED BY 'airflow';
GRANT ALL PRIVILEGES ON airflow.* TO 'airflow'@'%';

-- Use application database
USE depression_db;

-- ── Dimension Tables (Star Schema) ────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS dim_date (
    date_key    CHAR(8)     NOT NULL PRIMARY KEY,   -- YYYYMMDD
    year        SMALLINT    NOT NULL,
    month       TINYINT     NOT NULL,
    day         TINYINT     NOT NULL,
    quarter     TINYINT     NOT NULL,
    month_name  VARCHAR(12) GENERATED ALWAYS AS (
        CASE month
            WHEN 1  THEN 'January'  WHEN 2  THEN 'February' WHEN 3  THEN 'March'
            WHEN 4  THEN 'April'    WHEN 5  THEN 'May'       WHEN 6  THEN 'June'
            WHEN 7  THEN 'July'     WHEN 8  THEN 'August'    WHEN 9  THEN 'September'
            WHEN 10 THEN 'October'  WHEN 11 THEN 'November'  WHEN 12 THEN 'December'
        END
    ) STORED
) ENGINE=InnoDB;

CREATE TABLE IF NOT EXISTS dim_profession (
    profession_id   INT AUTO_INCREMENT PRIMARY KEY,
    profession_name VARCHAR(100) NOT NULL,
    industry        VARCHAR(100),
    sector          ENUM('Public', 'Private', 'Non-Profit', 'Self-Employed') DEFAULT 'Private'
) ENGINE=InnoDB;

CREATE TABLE IF NOT EXISTS dim_demographics (
    demographic_id  INT AUTO_INCREMENT PRIMARY KEY,
    age_group       VARCHAR(20),    -- e.g., '25-34'
    gender          VARCHAR(20),
    region          VARCHAR(100),
    education_level VARCHAR(50)
) ENGINE=InnoDB;

-- ── Fact Table ────────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS fact_depression_survey (
    survey_id                    BIGINT AUTO_INCREMENT PRIMARY KEY,
    date_key                     CHAR(8),
    profession_id                INT,
    demographic_id               INT,

    -- Features
    age                          TINYINT UNSIGNED,
    work_hours_per_week          DECIMAL(5,2),
    years_experience             TINYINT UNSIGNED,
    job_satisfaction             TINYINT UNSIGNED,
    sleep_hours                  DECIMAL(4,2),
    physical_activity_days       TINYINT UNSIGNED,
    social_interactions_per_week TINYINT UNSIGNED,
    has_mental_health_support    TINYINT(1) DEFAULT 0,
    remote_work                  TINYINT(1) DEFAULT 0,
    work_life_balance            DECIMAL(6,4),
    activity_sleep_ratio         DECIMAL(6,4),

    -- Label
    depressed                    TINYINT(1) NOT NULL,

    -- Metadata
    ingested_at                  DATETIME,
    loaded_at                    DATETIME DEFAULT CURRENT_TIMESTAMP,

    INDEX idx_date  (date_key),
    INDEX idx_label (depressed),
    FOREIGN KEY (date_key)       REFERENCES dim_date(date_key)       ON DELETE SET NULL,
    FOREIGN KEY (profession_id)  REFERENCES dim_profession(profession_id)  ON DELETE SET NULL,
    FOREIGN KEY (demographic_id) REFERENCES dim_demographics(demographic_id) ON DELETE SET NULL
) ENGINE=InnoDB;

-- ── Application Tables ─────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS predictions (
    id          BIGINT AUTO_INCREMENT PRIMARY KEY,
    input_data  JSON         NOT NULL,
    prediction  VARCHAR(50)  NOT NULL,
    probability DECIMAL(5,4) NOT NULL,
    created_at  DATETIME     DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_created (created_at),
    INDEX idx_prediction (prediction)
) ENGINE=InnoDB;

CREATE TABLE IF NOT EXISTS drift_reports (
    id               BIGINT AUTO_INCREMENT PRIMARY KEY,
    drift_score      DECIMAL(5,4)  NOT NULL,
    drift_detected   TINYINT(1)    NOT NULL,
    features_drifted JSON,
    method           VARCHAR(100),
    created_at       DATETIME      DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_created (created_at)
) ENGINE=InnoDB;

CREATE TABLE IF NOT EXISTS model_registry (
    id              BIGINT AUTO_INCREMENT PRIMARY KEY,
    model_version   VARCHAR(50)  NOT NULL,
    accuracy        DECIMAL(5,4),
    f1_score        DECIMAL(5,4),
    roc_auc         DECIMAL(5,4),
    model_path      VARCHAR(255),
    is_active       TINYINT(1)   DEFAULT 0,
    trained_at      DATETIME     DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_version (model_version),
    INDEX idx_active  (is_active)
) ENGINE=InnoDB;

-- ── Seed Dimension Data ───────────────────────────────────────────────────────

INSERT IGNORE INTO dim_profession (profession_name, industry, sector) VALUES
    ('Software Engineer',    'Technology',   'Private'),
    ('Data Scientist',       'Technology',   'Private'),
    ('Project Manager',      'Consulting',   'Private'),
    ('Financial Analyst',    'Finance',      'Private'),
    ('HR Manager',           'Human Resources', 'Private'),
    ('Marketing Specialist', 'Marketing',    'Private'),
    ('Teacher',              'Education',    'Public'),
    ('Nurse',                'Healthcare',   'Public');

INSERT IGNORE INTO dim_demographics (age_group, gender, region, education_level) VALUES
    ('18-24', 'Male',   'Kathmandu', 'Bachelor'),
    ('25-34', 'Female', 'Kathmandu', 'Master'),
    ('35-44', 'Male',   'Pokhara',   'Bachelor'),
    ('45-54', 'Female', 'Lalitpur',  'PhD'),
    ('55-64', 'Male',   'Bhaktapur', 'Bachelor');

FLUSH PRIVILEGES;
