from flask import Flask, jsonify, request
from flask_mysqldb import MySQL
from flask_cors import CORS
import datetime

app = Flask(__name__)

CORS(app)

app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = 'root'
app.config['MYSQL_DB'] = 'monitoramento_qml'
app.config['SECRET_KEY'] = 'sua-chave-secreta-aqui'

mysql = MySQL(app)


# A rota de login não precisa de alterações
@app.route('/api/login', methods=['POST'])
def login():
    data = request.get_json()
    email = data.get('email')
    senha_hash_do_cliente = data.get('senha')

    if not email or not senha_hash_do_cliente:
        return jsonify({'error': 'Email e senha são obrigatórios'}), 400

    try:
        cursor = mysql.connection.cursor()
        query = """
            SELECT id, email 
            FROM usuarios 
            WHERE email = %s AND senha_hash = %s
        """
        cursor.execute(query, (email, senha_hash_do_cliente))
        usuario = cursor.fetchone()
        cursor.close()

        if usuario:
            return jsonify({
                'message': 'Login bem-sucedido',
                'usuario_Id': str(usuario[0]),
                'usuario_email': str(usuario[1])
            })
        else:
            return jsonify({'error': 'Credenciais inválidas'}), 401

    except Exception as e:
        print(f"Erro no login: {e}")
        return jsonify({'error': 'Erro interno do servidor'}), 500


# --- ROTA DE ANÁLISES ATUALIZADA ---
@app.route('/api/analises', methods=['GET'])
def get_analises():
    """
    Busca os registros da tabela de análises, garantindo que as datas
    e os booleanos estejam em formatos corretos para JSON.
    """
    try:
        cursor = mysql.connection.cursor()
        query = """
            SELECT id, data_analise, acuracia, tp, tn, fp, fn, is_alerta 
            FROM qcyber_analises_vqc 
            ORDER BY data_analise DESC
        """
        cursor.execute(query)

        column_names = [desc[0] for desc in cursor.description]
        rows = cursor.fetchall()
        cursor.close()

        analises = []
        for row in rows:
            row_dict = dict(zip(column_names, row))

            # Garante que a data esteja no formato ISO 8601
            if 'data_analise' in row_dict and isinstance(row_dict['data_analise'], datetime.datetime):
                row_dict['data_analise'] = row_dict['data_analise'].isoformat()

            # <<< MUDANÇA CRÍTICA AQUI >>>
            # Converte o valor de 'is_alerta' (que vem como 0 ou 1) para um booleano.
            # A função bool() do Python converte 0 para False e 1 para True.
            if 'is_alerta' in row_dict:
                row_dict['is_alerta'] = bool(row_dict['is_alerta'])

            analises.append(row_dict)

        return jsonify(analises)

    except Exception as e:
        print(f"Erro ao buscar análises: {e}")
        return jsonify({'error': 'Erro ao buscar dados do banco'}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
