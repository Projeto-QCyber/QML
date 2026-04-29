# -*- coding: utf-8 -*-
"""
Script final e completo para recriar o banco de dados do projeto qCyber.
Esta versão une a estrutura original com as novas funcionalidades de autenticação.
"""
import os
import pymysql
from dotenv import load_dotenv


def recreate_database():
    """Recria o banco de dados qcyber_db completamente."""

    load_dotenv()
    db_name = os.getenv('MYSQL_DB', 'qcyber_db')

    print("=" * 60)
    print("CRIAÇÃO DA ESTRUTURA DO BANCO DE DADOS - PROJETO QCYBER")
    print("=" * 60)

    try:
        conn = pymysql.connect(
            host=os.getenv('MYSQL_HOST') or 'mysql',
            port=int(os.getenv('MYSQL_PORT') or 50000),
            user=os.getenv('MYSQL_USER') or os.getenv('MYSQL_USERNAME') or 'root',
            password=os.getenv('MYSQL_PASSWORD') or os.getenv('MYSQL_ROOT_PASSWORD') or 'root',
            charset='utf8mb4'
        )
        print("✅ Conectado ao servidor MySQL com sucesso!")
    except Exception as e:
        print(f"❌ ERRO: Falha ao conectar ao MySQL: {e}")
        return False

    try:
        with conn.cursor() as cursor:
            print(f"\nPASSO 1: Excluindo banco de dados '{db_name}'...")
            cursor.execute(f"DROP DATABASE IF EXISTS {db_name}")
            print(f"✅ Banco '{db_name}' excluído (se existia).")

            print(f"\nPASSO 2: Criando novo banco de dados '{db_name}'...")
            cursor.execute(f"CREATE DATABASE {db_name} CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci")
            print(f"✅ Banco '{db_name}' criado com sucesso.")

            cursor.execute(f"USE {db_name}")

            print("\nPASSO 3: Criando e populando tabelas de Lookup (ENUMs) e Configurações...")

            # Tabela para Tipos de Ataque
            cursor.execute("""
                           CREATE TABLE enum_tipo_ataque
                           (
                               id        INT PRIMARY KEY,
                               nome      VARCHAR(100) NOT NULL UNIQUE,
                               descricao TEXT
                           ) ENGINE=InnoDB;
                           """)
            label_map = {
                'Backdoor': 0, 'DDoS_HTTP': 1, 'DDoS_ICMP': 2, 'DDoS_TCP': 3, 'DDoS_UDP': 4,
                'Fingerprinting': 5, 'MITM': 6, 'Password': 7, 'Port_Scanning': 8, 'Ransomware': 9,
                'SQL_injection': 10, 'Uploading': 11, 'Vulnerability_scanner': 12, 'XSS': 13, 'Normal': 99
            }
            for nome, id_ataque in label_map.items():
                cursor.execute("INSERT INTO enum_tipo_ataque (id, nome, descricao) VALUES (%s, %s, %s)",
                               (id_ataque, nome, f'Detecção do tipo {nome}.'))
            print("✅ Tabela 'enum_tipo_ataque' criada e populada.")

            # Tabela para Status de Resposta
            cursor.execute("""
                           CREATE TABLE enum_status_resposta
                           (
                               id        INT AUTO_INCREMENT PRIMARY KEY,
                               nome      VARCHAR(100) NOT NULL UNIQUE,
                               descricao TEXT
                           ) ENGINE=InnoDB;
                           """)
            status_resposta = [
                ('Pendente', 'Detecção aguardando triagem.'),
                ('Ação Automática Executada', 'Sistema executou uma ação de contenção.'),
                ('Análise Manual Necessária', 'Requer investigação de um analista.'),
                ('Ignorado', 'Marcado como falso positivo ou de baixo risco.')
            ]
            cursor.executemany("INSERT INTO enum_status_resposta (nome, descricao) VALUES (%s, %s)", status_resposta)
            print("✅ Tabela 'enum_status_resposta' criada e populada.")

            # Tabela para Status de Incidente
            cursor.execute("""
                           CREATE TABLE enum_status_incidente
                           (
                               id        INT AUTO_INCREMENT PRIMARY KEY,
                               nome      VARCHAR(100) NOT NULL UNIQUE,
                               descricao TEXT
                           ) ENGINE=InnoDB;
                           """)
            status_incidente = [
                ('Aberto', 'Incidente recém-criado, aguardando análise inicial.'),
                ('Em Análise', 'Incidente está sendo ativamente investigado.'),
                ('Resolvido', 'Ameaça contida e incidente concluído.'),
                ('Ignorado', 'Incidente avaliado e considerado não-crítico ou falso positivo.')
            ]
            cursor.executemany("INSERT INTO enum_status_incidente (nome, descricao) VALUES (%s, %s)", status_incidente)
            print("✅ Tabela 'enum_status_incidente' criada e populada.")

            # Tabela para Nível de Risco
            cursor.execute("""
                           CREATE TABLE enum_nivel_risco
                           (
                               id        INT AUTO_INCREMENT PRIMARY KEY,
                               nome      VARCHAR(100) NOT NULL UNIQUE,
                               descricao TEXT
                           ) ENGINE=InnoDB;
                           """)
            niveis_risco = [
                ('Baixo', 'Impacto mínimo, geralmente informativo.'),
                ('Médio', 'Requer atenção, mas não é uma ameaça imediata.'),
                ('Alto', 'Ameaça significativa que pode impactar os serviços.'),
                ('Crítico', 'Ameaça grave com impacto iminente ou em andamento.')
            ]
            cursor.executemany("INSERT INTO enum_nivel_risco (nome, descricao) VALUES (%s, %s)", niveis_risco)
            print("✅ Tabela 'enum_nivel_risco' criada e populada.")

            # Tabela para Status de Dispositivo
            cursor.execute("""
                           CREATE TABLE enum_status_dispositivo
                           (
                               id        INT AUTO_INCREMENT PRIMARY KEY,
                               nome      VARCHAR(100) NOT NULL UNIQUE,
                               descricao TEXT
                           ) ENGINE=InnoDB;
                           """)
            status_dispositivo = [
                ('Ativo', 'Dispositivo online e monitorado.'),
                ('Inativo', 'Dispositivo offline.'),
                ('Em Manutenção', 'Dispositivo temporariamente fora de monitoramento para manutenção.')
            ]
            cursor.executemany("INSERT INTO enum_status_dispositivo (nome, descricao) VALUES (%s, %s)",
                               status_dispositivo)
            print("✅ Tabela 'enum_status_dispositivo' criada e populada.")

            # Tabela para Ações Executadas
            cursor.execute("""
                           CREATE TABLE enum_acao_executada
                           (
                               id        INT AUTO_INCREMENT PRIMARY KEY,
                               nome      VARCHAR(100) NOT NULL UNIQUE,
                               descricao TEXT
                           ) ENGINE=InnoDB;
                           """)
            acoes_executadas = [
                ('BLOCK_IP', 'Bloqueia um endereço de IP específico na firewall.'),
                ('ISOLATE_HOST', 'Coloca o dispositivo em uma rede de quarentena.'),
                ('DISABLE_USER', 'Desabilita uma conta de usuário no sistema.'),
                ('TERMINATE_PROCESS', 'Finaliza um processo malicioso específico pelo seu ID ou nome.')
            ]
            cursor.executemany("INSERT INTO enum_acao_executada (nome, descricao) VALUES (%s, %s)", acoes_executadas)
            print("✅ Tabela 'enum_acao_executada' criada e populada.")

            # Tabela de Configurações do Sistema
            cursor.execute("""
                           CREATE TABLE configuracoes
                           (
                               chave VARCHAR(50) PRIMARY KEY,
                               valor VARCHAR(255) NOT NULL
                           ) ENGINE=InnoDB;
                           """)
            configuracoes_padrao = [
                ('SMTP_SERVER', 'smtp.example.com'),
                ('SMTP_PORT', '587'),
                ('SMTP_USER', 'user@example.com'),
                ('SMTP_PASSWORD', 'password'),
                ('SMTP_SENDER_NAME', 'qCyber Platform')
            ]
            cursor.executemany("INSERT INTO configuracoes (chave, valor) VALUES (%s, %s)", configuracoes_padrao)
            print("✅ Tabela 'configuracoes' criada e populada com valores padrão.")

            conn.commit()

            print("\nPASSO 4: Criando tabelas principais com chaves estrangeiras...")

            cursor.execute("""
                           CREATE TABLE usuarios
                           (
                               id                           INT AUTO_INCREMENT PRIMARY KEY,
                               nome                         VARCHAR(100) NOT NULL,
                               email                        VARCHAR(100) NOT NULL UNIQUE,
                               senha_hash                   VARCHAR(255) NOT NULL,
                               ativo                        BOOLEAN   DEFAULT FALSE,
                               is_admin                     BOOLEAN   DEFAULT FALSE,
                               tem_permissao_sistema        BOOLEAN   DEFAULT TRUE,
                               email_verificado             BOOLEAN   DEFAULT FALSE,
                               dois_fatores_ativo           BOOLEAN   DEFAULT FALSE,

                               -- Campos para verificação de conta e 2FA
                               codigo_verificacao           VARCHAR(255) NULL,
                               codigo_verificacao_expiracao TIMESTAMP NULL,
                               tentativas_verificacao       INT       DEFAULT 0,

                               -- NOVOS CAMPOS PARA RESET DE SENHA
                               reset_senha_token            VARCHAR(255) NULL,
                               reset_senha_expiracao        TIMESTAMP NULL,

                               data_criacao                 TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                               INDEX                        idx_email (email)
                           ) ENGINE=InnoDB
                           """)

            print("✅ Tabela 'usuarios' criada.")

            cursor.execute("""
                           CREATE TABLE log_atividades_usuarios
                           (
                               id              INT AUTO_INCREMENT PRIMARY KEY,
                               data_ocorrencia TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                               usuario_ator_id INT         NOT NULL,
                               usuario_alvo_id INT         NOT NULL,
                               tipo_acao       VARCHAR(50) NOT NULL,
                               detalhes        TEXT,
                               FOREIGN KEY (usuario_ator_id) REFERENCES usuarios (id),
                               FOREIGN KEY (usuario_alvo_id) REFERENCES usuarios (id)
                           ) ENGINE=InnoDB
                           """)
            print("✅ Tabela 'log_atividades_usuarios' criada.")

            cursor.execute("""
                           CREATE TABLE qcyber_analises_vqc
                           (
                               id           INT AUTO_INCREMENT PRIMARY KEY,
                               data_analise TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                               acuracia     FLOAT,
                               tp           INT,
                               tn           INT,
                               fp           INT,
                               fn           INT,
                               is_alerta    BOOLEAN   DEFAULT FALSE
                           ) ENGINE=InnoDB
                           """)
            print("✅ Tabela 'qcyber_analises_vqc' (legada) mantida.")

            cursor.execute("""
                           CREATE TABLE dispositivos
                           (
                               id            INT AUTO_INCREMENT PRIMARY KEY,
                               nome          VARCHAR(100) NOT NULL,
                               host          VARCHAR(100) NOT NULL UNIQUE,
                               localizacao   VARCHAR(255),
                               status_id     INT,
                               data_cadastro TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                               FOREIGN KEY (status_id) REFERENCES enum_status_dispositivo (id)
                           ) ENGINE=InnoDB
                           """)
            print("✅ Tabela 'dispositivos' (normalizada) criada.")

            cursor.execute("""
                           CREATE TABLE incidentes_analisados
                           (
                               id                 INT AUTO_INCREMENT PRIMARY KEY,
                               data_criacao       TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                               titulo             VARCHAR(255) NOT NULL,
                               status_id          INT,
                               dispositivo_id     INT          NOT NULL,
                               nivel_risco_id     INT,
                               data_deteccao      DATETIME     NOT NULL,
                               resumo_tecnico     TEXT         NOT NULL,
                               explicacao_llm     TEXT,
                               acoes_recomendadas JSON,
                               FOREIGN KEY (status_id) REFERENCES enum_status_incidente (id),
                               FOREIGN KEY (nivel_risco_id) REFERENCES enum_nivel_risco (id),
                               FOREIGN KEY (dispositivo_id) REFERENCES dispositivos (id) ON DELETE CASCADE
                           ) ENGINE=InnoDB
                           """)
            print("✅ Tabela 'incidentes_analisados' (normalizada) criada.")

            cursor.execute("""
                           CREATE TABLE deteccoes
                           (
                               id                  INT AUTO_INCREMENT PRIMARY KEY,
                               data_deteccao       TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                               dispositivo_id      INT NOT NULL,
                               predicao            INT NOT NULL,
                               tipo_ataque_id      INT NOT NULL,
                               relatorio_api       TEXT,
                               status_resposta_id  INT,
                               acao_executada_id   INT NULL COMMENT 'ID da ação da tabela enum_acao_executada',
                               acao_parametro      VARCHAR(255) NULL COMMENT 'Parâmetro para a ação. Ex: o IP a ser bloqueado',
                               data_acao_executada TIMESTAMP NULL,
                               incidente_id        INT NULL,
                               FOREIGN KEY (dispositivo_id) REFERENCES dispositivos (id) ON DELETE CASCADE,
                               FOREIGN KEY (tipo_ataque_id) REFERENCES enum_tipo_ataque (id),
                               FOREIGN KEY (status_resposta_id) REFERENCES enum_status_resposta (id),
                               FOREIGN KEY (acao_executada_id) REFERENCES enum_acao_executada (id),
                               FOREIGN KEY (incidente_id) REFERENCES incidentes_analisados (id) ON DELETE SET NULL
                           ) ENGINE=InnoDB
                           """)
            print("✅ Tabela 'deteccoes' (normalizada) criada.")

        conn.commit()
        print("\nTodas as alterações foram salvas no banco de dados!")

    except Exception as e:
        print(f"❌ ERRO durante a recriação: {e}")
        if 'conn' in locals() and conn.open:
            conn.rollback()
        return False
    finally:
        if 'conn' in locals() and conn.open:
            conn.close()

    if not verify_database():
        return False

    return True


def ensure_bootstrap():
    """Garante que o banco e o esquema mínimo existam.
    - Cria o DB se não existir.
    - Se tabelas críticas estiverem ausentes, chama recreate_database().
    Retorna True em sucesso, False caso contrário.
    """
    load_dotenv()
    db_name = os.getenv('MYSQL_DB', 'qcyber_db')
    app_env = (os.getenv("APP_ENV") or os.getenv("FLASK_ENV") or "development").strip().lower()
    try:
        _host = os.getenv('MYSQL_HOST') or 'mysql'
        _user = os.getenv('MYSQL_USER') or os.getenv('MYSQL_USERNAME') or 'root'
        _pass = os.getenv('MYSQL_PASSWORD') or os.getenv('MYSQL_ROOT_PASSWORD') or 'root'
        print(f"[DB-BS] Conectando para bootstrap host={_host} user={_user}")
        conn = pymysql.connect(
            host=_host,
            user=_user,
            password=_pass,
            charset='utf8mb4'
        )
        with conn.cursor() as cursor:
            # Toma um lock para evitar corrida entre múltiplos workers
            cursor.execute("SELECT GET_LOCK(%s, 30)", ("qcyber_bootstrap_lock",))
            got_lock = cursor.fetchone()[0] == 1
            try:
                # Garante existência do DB
                cursor.execute(f"CREATE DATABASE IF NOT EXISTS {db_name} CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci")
                cursor.execute(f"USE {db_name}")

                # Verifica se tabelas críticas existem
                cursor.execute(
                    "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema=%s AND table_name IN ("\
                    "'enum_status_resposta','enum_tipo_ataque','dispositivos','deteccoes')",
                    (db_name,)
                )
                critical_present = cursor.fetchone()[0]

                if critical_present < 4 and got_lock:
                    if app_env in {"production", "prod"}:
                        print("❌ Estrutura crítica ausente e APP_ENV=production. Recusando recriar banco automaticamente.")
                        cursor.execute("SELECT RELEASE_LOCK(%s)", ("qcyber_bootstrap_lock",))
                        got_lock = False
                        conn.close()
                        return False
                    print("⚠️ Estrutura crítica ausente. Recriando banco de dados (ambiente de dev/test)...")
                    # Libera lock antes de recriar para evitar reentrância após drop
                    cursor.execute("SELECT RELEASE_LOCK(%s)", ("qcyber_bootstrap_lock",))
                    got_lock = False
                    conn.close()
                    return recreate_database()

                # Verificações/patches pontuais de colunas críticas
                # 1) deteccoes.predicao
                cursor.execute(
                    "SELECT COUNT(*) FROM information_schema.columns "
                    "WHERE table_schema=%s AND table_name='deteccoes' AND column_name='predicao'",
                    (db_name,)
                )
                pred_col_present = cursor.fetchone()[0] == 1
                if not pred_col_present and got_lock:
                    print("⚠️ Coluna 'predicao' ausente em 'deteccoes'. Aplicando patch de schema...")
                    cursor.execute("ALTER TABLE deteccoes ADD COLUMN predicao INT NOT NULL DEFAULT 1 AFTER dispositivo_id")

                conn.commit()
            finally:
                if got_lock:
                    cursor.execute("SELECT RELEASE_LOCK(%s)", ("qcyber_bootstrap_lock",))
        conn.close()
        return True
    except Exception as e:
        print(f"❌ Falha ao garantir bootstrap do banco: {e}")
        return False


def verify_database():
    """Verifica se o banco de dados e as tabelas foram criados corretamente."""
    print("\nPASSO 5: Verificando a estrutura do banco de dados...")

    load_dotenv()
    db_name = os.getenv('MYSQL_DB', 'qcyber_db')

    try:
        conn = pymysql.connect(
            host=os.getenv('MYSQL_HOST') or 'mysql',
            port=int(os.getenv('MYSQL_PORT') or 50000),
            user=os.getenv('MYSQL_USER') or os.getenv('MYSQL_USERNAME') or 'root',
            password=os.getenv('MYSQL_PASSWORD') or os.getenv('MYSQL_ROOT_PASSWORD') or 'root',
            database=db_name,
            charset='utf8mb4'
        )
        with conn.cursor() as cursor:
            cursor.execute("SHOW TABLES")
            tables = [table[0] for table in cursor.fetchall()]

            print(f"\n✅ Verificação concluída. Tabelas encontradas em '{db_name}':")
            for table in sorted(tables):
                print(f"  - {table}")

        conn.close()
        print("\n🎉 SUCESSO: ESTRUTURA DO BANCO DE DADOS RECRIADA COM SUCESSO!")
        return True
    except Exception as e:
        print(f"❌ ERRO na verificação: {e}")
        return False


if __name__ == '__main__':
    print("ATENÇÃO: Este script irá apagar e recriar a ESTRUTURA do banco de dados 'qcyber_db'!")
    print("         NENHUM usuário será criado.")
    resposta = input("Você tem certeza que deseja continuar? (s/N): ")

    if resposta.lower() in ['s', 'sim', 'y', 'yes']:
        if recreate_database():
            print("\n🚀 Estrutura do banco de dados pronta!")
    else:
        print("Operação cancelada pelo usuário.")
