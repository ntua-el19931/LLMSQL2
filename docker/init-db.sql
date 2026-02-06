-- LLMSQL2 Geography Database Initialization
-- This script runs automatically when the PostgreSQL container starts

-- Create tables
CREATE TABLE IF NOT EXISTS state (
    state_name TEXT PRIMARY KEY,
    population INTEGER,
    area INTEGER,
    country_name TEXT,
    capital TEXT,
    density INTEGER
);

CREATE TABLE IF NOT EXISTS city (
    city_name TEXT,
    population INTEGER,
    country_name TEXT,
    state_name TEXT REFERENCES state(state_name)
);

CREATE TABLE IF NOT EXISTS river (
    river_name TEXT,
    length INTEGER,
    country_name TEXT,
    traverse TEXT
);

CREATE TABLE IF NOT EXISTS lake (
    lake_name TEXT,
    area INTEGER,
    country_name TEXT,
    state_name TEXT
);

CREATE TABLE IF NOT EXISTS mountain (
    mountain_name TEXT,
    mountain_altitude INTEGER,
    country_name TEXT,
    state_name TEXT
);

CREATE TABLE IF NOT EXISTS border_info (
    state_name TEXT REFERENCES state(state_name),
    border TEXT
);

CREATE TABLE IF NOT EXISTS highlow (
    state_name TEXT,
    highest_elevation INTEGER,
    lowest_point TEXT,
    highest_point TEXT,
    lowest_elevation INTEGER
);

-- Insert sample data
INSERT INTO state VALUES 
    ('texas', 17000000, 691201, 'usa', 'austin', 25),
    ('california', 29760021, 163696, 'usa', 'sacramento', 182),
    ('colorado', 3307912, 104094, 'usa', 'denver', 32),
    ('new york', 17990455, 49109, 'usa', 'albany', 366),
    ('florida', 12937926, 58664, 'usa', 'tallahassee', 221),
    ('arizona', 3665228, 113909, 'usa', 'phoenix', 32),
    ('washington', 4866692, 68139, 'usa', 'olympia', 71)
ON CONFLICT (state_name) DO NOTHING;

INSERT INTO city VALUES 
    ('austin', 345496, 'usa', 'texas'),
    ('houston', 1595138, 'usa', 'texas'),
    ('dallas', 904078, 'usa', 'texas'),
    ('sacramento', 275741, 'usa', 'california'),
    ('los angeles', 2966850, 'usa', 'california'),
    ('san francisco', 678974, 'usa', 'california'),
    ('denver', 492365, 'usa', 'colorado'),
    ('albany', 101727, 'usa', 'new york'),
    ('new york', 7071639, 'usa', 'new york'),
    ('tallahassee', 81548, 'usa', 'florida'),
    ('miami', 346865, 'usa', 'florida'),
    ('phoenix', 789704, 'usa', 'arizona'),
    ('tucson', 330537, 'usa', 'arizona'),
    ('seattle', 493846, 'usa', 'washington'),
    ('olympia', 42514, 'usa', 'washington');

INSERT INTO river VALUES 
    ('colorado', 1450, 'usa', 'colorado'),
    ('rio grande', 1885, 'usa', 'texas'),
    ('mississippi', 2340, 'usa', 'minnesota'),
    ('columbia', 1243, 'usa', 'washington');

INSERT INTO mountain VALUES
    ('mount rainier', 14411, 'usa', 'washington'),
    ('mount whitney', 14505, 'usa', 'california'),
    ('pikes peak', 14115, 'usa', 'colorado');

GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO postgres;

-- ============================================================
-- ADVISING DATABASE
-- ============================================================
\c postgres
DROP DATABASE IF EXISTS advising;
CREATE DATABASE advising;
\c advising

CREATE TABLE area (
    course_id INTEGER,
    area VARCHAR(100)
);

CREATE TABLE comment_instructor (
    instructor_id INTEGER,
    student_id INTEGER,
    score INTEGER,
    comment_text TEXT,
    PRIMARY KEY (instructor_id, student_id)
);

CREATE TABLE course (
    course_id INTEGER PRIMARY KEY,
    name VARCHAR(255),
    department VARCHAR(255),
    number VARCHAR(50),
    credits VARCHAR(20),
    advisory_requirement TEXT,
    enforced_requirement TEXT,
    description TEXT,
    num_semesters INTEGER,
    num_enrolled INTEGER,
    has_discussion VARCHAR(1),
    has_lab VARCHAR(1),
    has_projects VARCHAR(1),
    has_exams VARCHAR(1),
    num_reviews INTEGER,
    clarity_score INTEGER,
    easiness_score INTEGER,
    helpfulness_score INTEGER
);

CREATE TABLE instructor (
    instructor_id INTEGER PRIMARY KEY,
    name VARCHAR(255),
    uniqname VARCHAR(100)
);

CREATE TABLE offering_instructor (
    offering_instructor_id INTEGER PRIMARY KEY,
    offering_id INTEGER,
    instructor_id INTEGER
);

CREATE TABLE course_offering (
    offering_id INTEGER PRIMARY KEY,
    course_id INTEGER,
    semester INTEGER,
    section_number INTEGER,
    start_time TIME,
    end_time TIME,
    monday VARCHAR(1),
    tuesday VARCHAR(1),
    wednesday VARCHAR(1),
    thursday VARCHAR(1),
    friday VARCHAR(1),
    saturday VARCHAR(1),
    sunday VARCHAR(1),
    has_final_project VARCHAR(1),
    has_final_exam VARCHAR(1),
    textbook_url TEXT,
    class_address VARCHAR(255),
    allows_audit VARCHAR(1)
);

CREATE TABLE student (
    student_id INTEGER PRIMARY KEY,
    lastname VARCHAR(100),
    firstname VARCHAR(100),
    program_id INTEGER,
    declare_major VARCHAR(100),
    total_credit INTEGER,
    total_gpa DECIMAL(3,2),
    entered_as VARCHAR(50),
    admit_term INTEGER,
    predicted_graduation_semester INTEGER,
    degree VARCHAR(50),
    minor VARCHAR(100),
    internship VARCHAR(100)
);

CREATE TABLE program (
    program_id INTEGER PRIMARY KEY,
    name VARCHAR(255),
    college VARCHAR(100),
    introduction TEXT
);

CREATE TABLE student_record (
    student_id INTEGER,
    course_id INTEGER,
    semester INTEGER,
    grade VARCHAR(5),
    how VARCHAR(50),
    transfer_source VARCHAR(100),
    earn_credit VARCHAR(10),
    repeat_term VARCHAR(50),
    test_id VARCHAR(50)
);

CREATE TABLE semester (
    semester_id INTEGER PRIMARY KEY,
    semester VARCHAR(20),
    year INTEGER
);

CREATE TABLE program_requirement (
    program_id INTEGER,
    category VARCHAR(100),
    min_credit INTEGER,
    additional_req TEXT
);

CREATE TABLE course_prerequisite (
    pre_course_id INTEGER,
    course_id INTEGER
);

CREATE TABLE program_course (
    program_id INTEGER,
    course_id INTEGER,
    workload INTEGER,
    category VARCHAR(100)
);

CREATE TABLE course_tags_count (
    course_id INTEGER,
    clear_grading INTEGER,
    pop_quiz INTEGER,
    group_projects INTEGER,
    inspirational INTEGER,
    long_lectures INTEGER,
    extra_credit INTEGER,
    few_tests INTEGER,
    good_feedback INTEGER,
    tough_tests INTEGER,
    heavy_papers INTEGER,
    cares_for_students INTEGER,
    heavy_assignments INTEGER,
    respected INTEGER,
    participation INTEGER,
    heavy_reading INTEGER,
    tough_grader INTEGER,
    hilarious INTEGER,
    would_take_again INTEGER,
    good_lecture INTEGER,
    no_hierarchical INTEGER
);

CREATE TABLE gsi (
    course_offering_id INTEGER,
    student_id INTEGER
);

-- Sample data for advising
INSERT INTO course VALUES 
    (1001, 'Introduction to Programming', 'Computer Science', 'EECS 101', '4', NULL, NULL, 'Basic programming concepts', 2, 150, 'Y', 'Y', 'Y', 'Y', 45, 4, 3, 4),
    (1002, 'Data Structures', 'Computer Science', 'EECS 281', '4', 'EECS 101', NULL, 'Advanced data structures and algorithms', 2, 120, 'Y', 'N', 'Y', 'Y', 38, 3, 2, 4),
    (1003, 'Database Systems', 'Computer Science', 'EECS 484', '4', 'EECS 281', NULL, 'Relational database design and SQL', 1, 80, 'N', 'N', 'Y', 'Y', 25, 4, 3, 5);

INSERT INTO instructor VALUES 
    (101, 'Dr. Smith', 'dsmith'),
    (102, 'Dr. Johnson', 'djohnson'),
    (103, 'Dr. Williams', 'dwilliams');

INSERT INTO student VALUES 
    (1, 'Doe', 'John', 1, 'Computer Science', 60, 3.50, 'Freshman', 2023, 2027, 'BS', NULL, NULL),
    (2, 'Smith', 'Jane', 1, 'Computer Science', 45, 3.75, 'Freshman', 2024, 2028, 'BS', 'Math', NULL);

INSERT INTO semester VALUES 
    (1, 'Fall', 2024),
    (2, 'Winter', 2025),
    (3, 'Fall', 2025);

GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO postgres;

-- ============================================================
-- ATIS DATABASE (Airline Travel Information System)
-- ============================================================
\c postgres
DROP DATABASE IF EXISTS atis;
CREATE DATABASE atis;
\c atis

CREATE TABLE aircraft (
    aircraft_code VARCHAR(3),
    aircraft_description VARCHAR(100),
    manufacturer VARCHAR(50),
    basic_type VARCHAR(30),
    engines INTEGER,
    propulsion VARCHAR(20),
    wide_body VARCHAR(5),
    wing_span INTEGER,
    length INTEGER,
    weight INTEGER,
    capacity INTEGER,
    pay_load INTEGER,
    cruising_speed INTEGER,
    range_miles INTEGER,
    pressurized VARCHAR(5)
);

CREATE TABLE airline (
    airline_code VARCHAR(2),
    airline_name TEXT,
    note TEXT
);

CREATE TABLE airport (
    airport_code VARCHAR(3),
    airport_name TEXT,
    airport_location TEXT,
    state_code VARCHAR(2),
    country_name VARCHAR(10),
    time_zone_code VARCHAR(3),
    minimum_connect_time INTEGER
);

CREATE TABLE airport_service (
    city_code VARCHAR(4),
    airport_code VARCHAR(3),
    miles_distant INTEGER,
    direction VARCHAR(2),
    minutes_distant INTEGER
);

CREATE TABLE city (
    city_code VARCHAR(4),
    city_name VARCHAR(50),
    state_code VARCHAR(2),
    country_name VARCHAR(10),
    time_zone_code VARCHAR(3)
);

CREATE TABLE class_of_service (
    booking_class VARCHAR(2),
    rank INTEGER,
    class_description VARCHAR(100)
);

CREATE TABLE fare (
    fare_id INTEGER PRIMARY KEY,
    from_airport VARCHAR(3),
    to_airport VARCHAR(3),
    fare_basis_code VARCHAR(20),
    fare_airline VARCHAR(2),
    restriction_code VARCHAR(10),
    one_direction_cost DECIMAL(10,2),
    round_trip_cost DECIMAL(10,2),
    round_trip_required VARCHAR(5)
);

CREATE TABLE fare_basis (
    fare_basis_code VARCHAR(20),
    booking_class VARCHAR(2),
    class_type VARCHAR(20),
    premium VARCHAR(5),
    economy VARCHAR(5),
    discounted VARCHAR(5),
    night VARCHAR(5),
    season VARCHAR(10),
    basis_days VARCHAR(20)
);

CREATE TABLE flight (
    flight_id INTEGER PRIMARY KEY,
    flight_days VARCHAR(10),
    from_airport VARCHAR(3),
    to_airport VARCHAR(3),
    departure_time TIME,
    arrival_time TIME,
    airline_flight VARCHAR(20),
    airline_code VARCHAR(2),
    flight_number INTEGER,
    aircraft_code_sequence VARCHAR(20),
    meal_code VARCHAR(5),
    stops INTEGER,
    connections INTEGER,
    dual_carrier VARCHAR(5),
    time_elapsed INTEGER
);

CREATE TABLE flight_fare (
    flight_id INTEGER,
    fare_id INTEGER
);

CREATE TABLE flight_leg (
    flight_id INTEGER,
    leg_number INTEGER,
    leg_flight INTEGER
);

CREATE TABLE flight_stop (
    flight_id INTEGER,
    stop_number INTEGER,
    stop_days VARCHAR(10),
    stop_airport VARCHAR(3),
    arrival_time TIME,
    arrival_airline VARCHAR(2),
    arrival_flight_number INTEGER,
    departure_time TIME,
    departure_airline VARCHAR(2),
    departure_flight_number INTEGER,
    stop_time INTEGER
);

CREATE TABLE food_service (
    meal_code VARCHAR(5),
    meal_number INTEGER,
    compartment VARCHAR(20),
    meal_description VARCHAR(100)
);

CREATE TABLE ground_service (
    city_code VARCHAR(4),
    airport_code VARCHAR(3),
    transport_type VARCHAR(20),
    ground_fare DECIMAL(10,2)
);

CREATE TABLE restriction (
    restriction_code VARCHAR(10),
    advance_purchase INTEGER,
    stopovers VARCHAR(5),
    saturday_stay_required VARCHAR(5),
    minimum_stay INTEGER,
    maximum_stay INTEGER,
    application VARCHAR(50),
    no_discounts VARCHAR(5)
);

CREATE TABLE state (
    state_code VARCHAR(2),
    state_name VARCHAR(50),
    country_name VARCHAR(20)
);

CREATE TABLE time_zone (
    time_zone_code VARCHAR(3),
    time_zone_name VARCHAR(50),
    hours_from_gmt INTEGER
);

-- Sample ATIS data
INSERT INTO airline VALUES 
    ('AA', 'AMERICAN AIRLINES', NULL),
    ('UA', 'UNITED AIRLINES', NULL),
    ('DL', 'DELTA AIR LINES', NULL),
    ('NW', 'NORTHWEST AIRLINES', NULL),
    ('CO', 'CONTINENTAL AIRLINES', NULL);

INSERT INTO airport VALUES 
    ('ATL', 'HARTSFIELD ATLANTA INTL', 'ATLANTA, GA', 'GA', 'USA', 'EST', 55),
    ('BOS', 'LOGAN INTERNATIONAL', 'BOSTON, MA', 'MA', 'USA', 'EST', 40),
    ('DEN', 'DENVER INTERNATIONAL', 'DENVER, CO', 'CO', 'USA', 'MST', 50),
    ('DFW', 'DALLAS/FORT WORTH INTL', 'DALLAS, TX', 'TX', 'USA', 'CST', 55),
    ('JFK', 'JF KENNEDY INTERNATIONAL', 'NEW YORK, NY', 'NY', 'USA', 'EST', 60),
    ('LAX', 'LOS ANGELES INTERNATIONAL', 'LOS ANGELES, CA', 'CA', 'USA', 'PST', 70),
    ('ORD', 'OHARE INTERNATIONAL', 'CHICAGO, IL', 'IL', 'USA', 'CST', 50),
    ('SFO', 'SAN FRANCISCO INTL', 'SAN FRANCISCO, CA', 'CA', 'USA', 'PST', 50);

INSERT INTO city VALUES 
    ('MATL', 'ATLANTA', 'GA', 'USA', 'EST'),
    ('BBOS', 'BOSTON', 'MA', 'USA', 'EST'),
    ('DDEN', 'DENVER', 'CO', 'USA', 'MST'),
    ('DDFW', 'DALLAS', 'TX', 'USA', 'CST'),
    ('NNYC', 'NEW YORK', 'NY', 'USA', 'EST'),
    ('LLAX', 'LOS ANGELES', 'CA', 'USA', 'PST'),
    ('CCHI', 'CHICAGO', 'IL', 'USA', 'CST'),
    ('SSFO', 'SAN FRANCISCO', 'CA', 'USA', 'PST');

INSERT INTO flight VALUES 
    (1, 'DAILY', 'BOS', 'SFO', '08:00', '11:30', 'AA1234', 'AA', 1234, '757', 'B', 0, 0, 'NO', 330),
    (2, 'DAILY', 'JFK', 'LAX', '09:00', '12:00', 'UA5678', 'UA', 5678, '767', 'L', 0, 0, 'NO', 360),
    (3, 'DAILY', 'ORD', 'DEN', '14:00', '15:30', 'DL9012', 'DL', 9012, '737', 'S', 0, 0, 'NO', 150);

INSERT INTO aircraft VALUES 
    ('757', 'BOEING 757-200', 'BOEING', '757', 2, 'JET', 'NO', 125, 155, 220000, 239, NULL, 593, 3247, 'YES'),
    ('767', 'BOEING 767-300', 'BOEING', '767', 2, 'JET', 'YES', 156, 180, 300000, 290, 43200, 593, 3639, 'YES'),
    ('737', 'BOEING 737-800', 'BOEING', '737', 2, 'JET', 'NO', 118, 129, 174200, 189, 45000, 520, 2940, 'YES');

GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO postgres;

-- ============================================================
-- RESTAURANTS DATABASE
-- ============================================================
\c postgres
DROP DATABASE IF EXISTS restaurants;
CREATE DATABASE restaurants;
\c restaurants

CREATE TABLE geographic (
    city_name VARCHAR(100),
    county VARCHAR(100),
    region VARCHAR(50)
);

CREATE TABLE restaurant (
    id INTEGER PRIMARY KEY,
    name VARCHAR(255),
    food_type VARCHAR(100),
    city_name VARCHAR(100),
    rating DECIMAL(3,1)
);

CREATE TABLE location (
    restaurant_id INTEGER REFERENCES restaurant(id),
    house_number INTEGER,
    street_name VARCHAR(255),
    city_name VARCHAR(100)
);

-- Sample restaurants data
INSERT INTO geographic VALUES 
    ('san francisco', 'san francisco', 'bay area'),
    ('oakland', 'alameda', 'bay area'),
    ('berkeley', 'alameda', 'bay area'),
    ('palo alto', 'santa clara', 'bay area'),
    ('san jose', 'santa clara', 'bay area'),
    ('mountain view', 'santa clara', 'bay area'),
    ('los gatos', 'santa clara', 'bay area');

INSERT INTO restaurant VALUES 
    (1, 'zuni cafe', 'american', 'san francisco', 4.2),
    (2, 'tadich grill', 'seafood', 'san francisco', 4.0),
    (3, 'house of prime rib', 'american', 'san francisco', 4.5),
    (4, 'la taqueria', 'mexican', 'san francisco', 4.3),
    (5, 'hog island oyster', 'seafood', 'san francisco', 4.4),
    (6, 'chez panisse', 'french', 'berkeley', 4.6),
    (7, 'pizzaiolo', 'italian', 'oakland', 4.2),
    (8, 'commis', 'french', 'oakland', 4.5),
    (9, 'manresa', 'french', 'los gatos', 4.7),
    (10, 'benu', 'asian fusion', 'san francisco', 4.8);

INSERT INTO location VALUES 
    (1, 1658, 'market street', 'san francisco'),
    (2, 240, 'california street', 'san francisco'),
    (3, 1906, 'van ness avenue', 'san francisco'),
    (4, 2889, 'mission street', 'san francisco'),
    (5, 1, 'ferry building', 'san francisco'),
    (6, 1517, 'shattuck avenue', 'berkeley'),
    (7, 5008, 'telegraph avenue', 'oakland'),
    (8, 3859, 'piedmont avenue', 'oakland'),
    (9, 320, 'village lane', 'los gatos'),
    (10, 22, 'hawthorne street', 'san francisco');

GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO postgres;
