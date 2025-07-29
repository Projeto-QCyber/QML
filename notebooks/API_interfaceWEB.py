from flask import Flask, jsonify, request
from flask_mysqldb import MySQL
from flask_cors import CORS
import datetime

app = Flask(__name__)
CORS(app)

# --- Configurações (sem alterações) ---
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = 'root'
app.config['MYSQL_DB'] = 'monitoramento_qml'
app.config['SECRET_KEY'] = 'sua-chave-secreta-aqui'

mysql = MySQL(app)


# --- Rota de Login (sem alterações) ---
@app.route('/api-interface/login', methods=['POST'])
def login():
    # ... (código existente sem alterações)
    data = request.get_json()
    email = data.get('email')
    senha_hash_do_cliente = data.get('senha')
    if not email or not senha_hash_do_cliente:
        return jsonify({'error': 'Email e senha são obrigatórios'}), 400
    try:
        cursor = mysql.connection.cursor()
        query = "SELECT id, email FROM usuarios WHERE email = %s AND senha_hash = %s"
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


# --- ROTA DE KPIs ATUALIZADA (com filtro de data) ---
@app.route('/api-interface/dashboard-kpis', methods=['GET'])
def get_dashboard_kpis():
    """
    Calcula e retorna as métricas chave (KPIs) para o topo do dashboard.
    """
    try:
        cursor = mysql.connection.cursor()

        # KPI 1: Total de Alertas nas últimas 24 horas
        query_alerts = """
            SELECT COUNT(id) 
            FROM qcyber_analises_vqc 
            WHERE is_alerta = 1 AND data_analise >= NOW() - INTERVAL 1 DAY
        """
        cursor.execute(query_alerts)
        total_alerts_24h = cursor.fetchone()[0]

        # KPI 2: Acurácia Média nas últimas 24 horas
        query_accuracy = """
            SELECT AVG(acuracia) 
            FROM qcyber_analises_vqc 
            WHERE data_analise >= NOW() - INTERVAL 1 DAY
        """
        cursor.execute(query_accuracy)
        avg_accuracy_24h = cursor.fetchone()[0]

        # KPI 3: Ameaças Bloqueadas (soma de Verdadeiros Positivos nas últimas 24h)
        query_threats = """
            SELECT SUM(tp)
            FROM qcyber_analises_vqc
            WHERE data_analise >= NOW() - INTERVAL 1 DAY
        """
        cursor.execute(query_threats)
        ameacas_bloqueadas = cursor.fetchone()[0]

        # <<< MUDANÇA AQUI >>>
        # KPI 4: Dispositivos Protegidos (contagem de dispositivos ativos)
        query_devices = "SELECT COUNT(id) FROM dispositivos WHERE status = 'Ativo'"
        cursor.execute(query_devices)
        dispositivos_protegidos = cursor.fetchone()[0]

        cursor.close()

        return jsonify({
            "total_alerts_24h": total_alerts_24h or 0,
            "avg_accuracy_24h": float(avg_accuracy_24h or 0),
            "dispositivos_protegidos": dispositivos_protegidos or 0,
            "ameacas_bloqueadas": int(ameacas_bloqueadas or 0)
        })

    except Exception as e:
        print(f"Erro ao buscar KPIs: {e}")
        return jsonify({'error': 'Erro ao buscar dados para os KPIs.'}), 500


# --- ROTA DE ANÁLISES ATUALIZADA (com filtro de data) ---
@app.route('/api-interface/analises', methods=['GET'])
def get_analises():
    """
    Busca os registros da tabela de análises com suporte a filtros e paginação.
    Query Params: ?page=1&limit=10&filter=alerts_only&start_date=...&end_date=...
    """
    try:
        page = request.args.get('page', 1, type=int)
        limit = request.args.get('limit', 10, type=int)
        filter_option = request.args.get('filter', 'all', type=str)
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        offset = (page - 1) * limit

        base_query = "FROM qcyber_analises_vqc"
        where_clauses = []
        params = []

        if filter_option == 'alerts_only':
            where_clauses.append("is_alerta = 1")

        if start_date and end_date:
            end_date_inclusive = datetime.datetime.strptime(end_date, '%Y-%m-%d') + datetime.timedelta(days=1)
            where_clauses.append("data_analise BETWEEN %s AND %s")
            params.extend([start_date, end_date_inclusive.strftime('%Y-%m-%d')])

        if where_clauses:
            base_query += " WHERE " + " AND ".join(where_clauses)

        cursor = mysql.connection.cursor()

        cursor.execute(f"SELECT COUNT(id) {base_query}", params)
        total_items = cursor.fetchone()[0]

        data_query = f"""
            SELECT id, data_analise, acuracia, tp, tn, fp, fn, is_alerta 
            {base_query} 
            ORDER BY data_analise DESC 
            LIMIT %s OFFSET %s
        """
        final_params = params + [limit, offset]
        cursor.execute(data_query, final_params)

        column_names = [desc[0] for desc in cursor.description]
        rows = cursor.fetchall()
        cursor.close()

        analises = []
        for row in rows:
            row_dict = dict(zip(column_names, row))
            if 'data_analise' in row_dict and isinstance(row_dict['data_analise'], datetime.datetime):
                row_dict['data_analise'] = row_dict['data_analise'].isoformat()
            if 'is_alerta' in row_dict:
                row_dict['is_alerta'] = bool(row_dict['is_alerta'])
            analises.append(row_dict)

        return jsonify({
            "total_items": total_items,
            "page": page,
            "limit": limit,
            "data": analises
        })

    except Exception as e:
        print(f"Erro ao buscar análises: {e}")
        return jsonify({'error': 'Erro ao buscar dados do banco'}), 500


@app.route('/api-interface/dispositivos', methods=['GET'])
def get_dispositivos():
    """Busca e retorna a lista de todos os dispositivos cadastrados."""
    try:
        cursor = mysql.connection.cursor()
        cursor.execute("SELECT id, nome, host, localizacao, status, data_cadastro FROM dispositivos ORDER BY nome ASC")

        column_names = [desc[0] for desc in cursor.description]
        dispositivos = [dict(zip(column_names, row)) for row in cursor.fetchall()]
        cursor.close()

        # Formata a data para o padrão ISO 8601
        for dispositivo in dispositivos:
            if 'data_cadastro' in dispositivo and isinstance(dispositivo['data_cadastro'], datetime.datetime):
                dispositivo['data_cadastro'] = dispositivo['data_cadastro'].isoformat()

        return jsonify(dispositivos)
    except Exception as e:
        print(f"Erro ao buscar dispositivos: {e}")
        return jsonify({'error': 'Erro ao buscar dados dos dispositivos.'}), 500


@app.route('/api-interface/dispositivos', methods=['POST'])
def create_dispositivo():
    """Cadastra um novo dispositivo no banco de dados."""
    data = request.get_json()
    nome = data.get('nome')
    host = data.get('host')
    localizacao = data.get('localizacao')

    if not nome or not host:
        return jsonify({'error': 'Nome e Host são campos obrigatórios.'}), 400

    try:
        cursor = mysql.connection.cursor()
        # Verifica se o host já existe para evitar duplicados
        cursor.execute("SELECT id FROM dispositivos WHERE host = %s", (host,))
        if cursor.fetchone():
            return jsonify({'error': 'Este host já está cadastrado.'}), 409  # 409 Conflict

        cursor.execute(
            "INSERT INTO dispositivos (nome, host, localizacao) VALUES (%s, %s, %s)",
            (nome, host, localizacao)
        )
        mysql.connection.commit()
        new_id = cursor.lastrowid
        cursor.close()

        return jsonify({'message': 'Dispositivo cadastrado com sucesso!', 'id': new_id}), 201
    except Exception as e:
        mysql.connection.rollback()
        print(f"Erro ao cadastrar dispositivo: {e}")
        return jsonify({'error': 'Erro interno ao cadastrar o dispositivo.'}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
