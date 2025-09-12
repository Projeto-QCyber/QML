import pymysql
from ..utils.generic import get_env_var


def get_db_connection():
    """Cria e retorna uma nova conexão com o banco para cada requisição."""
    return pymysql.connect(
        host=get_env_var('MYSQL_HOST', 'localhost'),
        user=get_env_var('MYSQL_USER', 'root'),
        password=get_env_var('MYSQL_PASSWORD', 'root'),
        database=get_env_var('MYSQL_DATABASE', 'qcyberDB'),
        cursorclass=pymysql.cursors.DictCursor
    )


def get_lookup_ids(cursor, table_name):
    """ Busca IDs de uma tabela de lookup."""
    cursor.execute(f"SELECT id, nome FROM {table_name}")
    return {row['nome']: row['id'] for row in cursor.fetchall()}