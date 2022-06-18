# GUI tool to draw data bases:
# https://kitt.lewagon.com/db

# example using pandas

import sqlite3
import pandas as pd

conn = sqlite3.connect('data/soccer.sqlite')
query = '''
SELECT *
FROM table
'''
df = pd.read_sql(query, conn)

df.set_index() # permet de definir les index
df.index # permet d’accéder aux index

# this is how we pass parameters in order to avoid SQL injection
# (in this example, we don't use pandas)
conn = sqlite3.connect('data/exploitable_db.sqlite')
c = conn.cursor()
query = """
SELECT *
FROM users
WHERE users.username = ?
AND users.password = ?
"""
c.execute(query, (username, password))

# more SQL examples

# where
'''
SELECT matches.id, matches.season, matches.stage, matches.date
FROM "Match" AS matches
WHERE matches.country_id = 1 OR matches.country_id = 1729
'''

'''
SELECT matches.id, matches.season, matches.stage, matches.date
FROM "Match" AS matches
WHERE matches.country_id IN (1, 1729)
'''

'''
SELECT *
FROM Player
WHERE UPPER(Player.player_name) LIKE 'JOHN %'
'''

# count
'''
SELECT COUNT(Player.id)
FROM Player
WHERE Player.height >= 200
'''

# order by
'''
SELECT *
FROM Player
ORDER BY Player.weight DESC
LIMIT 10
'''

# group by
'''
SELECT COUNT(matches.id) AS match_count, matches.country_id
FROM "Match" AS matches
GROUP BY matches.country_id
HAVING match_count >= 3000
ORDER BY match_count DESC
'''

# case (if then)
'''
SELECT
COUNT(matches.id) AS outcome_count,
CASE
    WHEN matches.home_team_goal > matches.away_team_goal
        THEN 'home_win'
    WHEN matches.home_team_goal = matches.away_team_goal
        THEN 'draw'
    ELSE 'away_win'
END AS outcome
FROM "Match" AS matches
GROUP BY outcome
ORDER BY outcome_count DESC
'''

# join
'''
SELECT League.name, Country.name
FROM League
LEFT JOIN Country ON League.country_id = Country.id
'''

# insert
'''
INSERT INTO table (column1, column2, ...)
VALUES(value1, value2 , ...)
'''

# update
'''
UPDATE table
SET column_1 = new_value_1,
    column_2 = new_value_2
WHERE
    search_condition
'''

'''
UPDATE Country
SET
    name = 'République Française'
WHERE
    id = 4769
'''

# delete
'''
DELETE FROM table
WHERE search_condition
'''

'''
DELETE FROM Country
WHERE id = 4769
'''

# string functions
'''
SUBSTR
INSTR
TRIM (LTRIM, RTRIM)
LENGTH
UPPER / LOWER
REPLACE
'''

# rank
'''
SELECT
    orders.id,
    orders.ordered_at,
    orders.customer_id,
    RANK() OVER (
        PARTITION BY orders.customer_id
        ORDER BY orders.ordered_at
    ) AS order_rank
FROM orders
'''

# sum
'''
SELECT
    orders.id,
    orders.ordered_at,
    orders.amount,
    orders.customer_id,
    SUM(orders.amount) OVER (
        PARTITION BY orders.customer_id
        ORDER BY orders.ordered_at
    ) AS cumulative_amount
FROM orders
'''

# with

'''
WITH matches_per_month AS (
    SELECT
        STRFTIME('%Y-%m', DATE(matches.date)) AS period,
        COUNT(*) AS cnt
    FROM "Match" AS matches
    GROUP BY period
    ORDER BY period
)
SELECT
    matches_per_month.period,
    SUM(matches_per_month.cnt) OVER (
        ORDER BY matches_per_month.period
    ) AS cumulative_count
FROM matches_per_month
'''
