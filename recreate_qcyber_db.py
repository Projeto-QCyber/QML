# -*- coding: utf-8 -*-
"""
Script para recriar o banco de dados do projeto qCyber.

Este script executa as seguintes ações:
1.  Conecta-se ao servidor MySQL usando credenciais de um arquivo .env.
2.  Exclui (DROP) o banco de dados 'qcyberDB' se ele já existir.
3.  Cria (CREATE) um novo banco de dados 'qcyberDB' com codificação UTF-8.
4.  Cria as tabelas:
    - usuarios: com todos os campos de gestão.
    - dispositivos: para monitoramento de hosts.
    - log_atividades_usuarios: para auditoria de ações.
    - qcyber_analises_vqc: (Mantida) Tabela legada para análises.
    - incidentes_analisados: Para relatórios detalhados de incidentes.
    - deteccoes_individuais: (NOVA) Estrutura granular para cada detecção de ataque.
5.  Verifica as tabelas e cria usuários padrão.
"""
import os
import sys
import pymysql
import hashlib
from dotenv import load_dotenv


def hash_password(password):
    """Gera o hash de uma senha usando SHA256."""
    return hashlib.sha256(password.encode('utf-8')).hexdigest()


def recreate_database():
    """Recria o banco de dados qcyberDB completamente."""

    load_dotenv()
    db_name = os.getenv('MYSQL_DB', 'qcyberDB')

    print("=" * 60)
    print("RECRIACAO DO BANCO DE DADOS - PROJETO QCYBER")
    print("=" * 60)

    try:
        conn = pymysql.connect(
            host=os.getenv('MYSQL_HOST', 'localhost'),
            user=os.getenv('MYSQL_USER', 'root'),
            password=os.getenv('MYSQL_PASSWORD', 'root'),
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

            print("\nPASSO 3: Criando tabelas...")

            # Tabelas de Entidades Principais
            cursor.execute("""
                           CREATE TABLE usuarios
                           (
                               id                    INT AUTO_INCREMENT PRIMARY KEY,
                               nome                  VARCHAR(100) NOT NULL,
                               email                 VARCHAR(100) NOT NULL UNIQUE,
                               senha_hash            VARCHAR(255) NOT NULL,
                               ativo                 BOOLEAN   DEFAULT TRUE,
                               is_admin              BOOLEAN   DEFAULT FALSE,
                               tem_permissao_sistema BOOLEAN   DEFAULT TRUE,
                               email_verificado      BOOLEAN   DEFAULT FALSE,
                               dois_fatores_ativo    BOOLEAN   DEFAULT FALSE,
                               data_criacao          TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                               INDEX                 idx_email (email)
                           ) ENGINE=InnoDB
                           """)
            print("✅ Tabela 'usuarios' criada.")

            cursor.execute("""
                           CREATE TABLE dispositivos
                           (
                               id            INT AUTO_INCREMENT PRIMARY KEY,
                               nome          VARCHAR(100) NOT NULL,
                               host          VARCHAR(100) NOT NULL UNIQUE,
                               localizacao   VARCHAR(255),
                               status        VARCHAR(50) DEFAULT 'Ativo',
                               data_cadastro TIMESTAMP   DEFAULT CURRENT_TIMESTAMP,
                               INDEX         idx_nome (nome)
                           ) ENGINE=InnoDB
                           """)
            print("✅ Tabela 'dispositivos' criada.")

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
                               FOREIGN KEY (usuario_alvo_id) REFERENCES usuarios (id),
                               INDEX           idx_data_ocorrencia (data_ocorrencia),
                               INDEX           idx_usuario_alvo (usuario_alvo_id)
                           ) ENGINE=InnoDB
                           """)
            print("✅ Tabela 'log_atividades_usuarios' criada.")

            # Tabela Legada
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
                               is_alerta    BOOLEAN   DEFAULT FALSE,
                               INDEX        idx_data_analise (data_analise)
                           ) ENGINE=InnoDB
                           """)
            print("✅ Tabela 'qcyber_analises_vqc' (legada) mantida.")

            # Tabela de Incidentes (deve ser criada antes da de detecções por causa da FK)
            cursor.execute("""
                           CREATE TABLE incidentes_analisados
                           (
                               id                 INT AUTO_INCREMENT PRIMARY KEY,
                               data_criacao       TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                               titulo             VARCHAR(255) NOT NULL,
                               status             ENUM('Aberto', 'Em Análise', 'Resolvido', 'Ignorado') NOT NULL DEFAULT 'Aberto',
                               dispositivo_id     INT          NOT NULL,
                               nivel_risco        ENUM('Baixo', 'Médio', 'Alto', 'Crítico') NOT NULL DEFAULT 'Médio',
                               data_deteccao      DATETIME     NOT NULL,
                               resumo_tecnico     TEXT         NOT NULL,
                               explicacao_llm     TEXT,
                               acoes_recomendadas JSON,
                               FOREIGN KEY (dispositivo_id) REFERENCES dispositivos (id) ON DELETE CASCADE,
                               INDEX              idx_status_risco (status, nivel_risco)
                           ) ENGINE=InnoDB
                           """)
            print("✅ Tabela 'incidentes_analisados' criada.")

            # NOVA ESTRUTURA PARA DETECÇÕES
            cursor.execute("""
                           CREATE TABLE deteccoes_individuais
                           (
                               id                 INT AUTO_INCREMENT PRIMARY KEY,
                               data_deteccao      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                               lote_uuid          VARCHAR(36)  NOT NULL COMMENT 'Agrupa todas as detecções de um mesmo lote de análise',
                               dispositivo_id     INT          NOT NULL,
                               predicao           INT          NOT NULL COMMENT 'Resultado binário da predição (ex: 1)',
                               tipo_ataque        VARCHAR(100) NOT NULL COMMENT 'Classificação do ataque (ex: MITM, Ransomware)',
                               relatorio_api_lote TEXT,
                               incidente_id       INT NULL,
                               FOREIGN KEY (dispositivo_id) REFERENCES dispositivos (id) ON DELETE CASCADE,
                               FOREIGN KEY (incidente_id) REFERENCES incidentes_analisados (id) ON DELETE SET NULL,
                               INDEX              idx_data_deteccao (data_deteccao),
                               INDEX              idx_tipo_ataque (tipo_ataque),
                               INDEX              idx_lote_uuid (lote_uuid)
                           ) ENGINE=InnoDB
                           """)
            print("✅ Tabela 'deteccoes_individuais' criada.")

        conn.commit()
        print("\nTodas as alterações foram salvas no banco de dados!")

    except Exception as e:
        print(f"❌ ERRO durante a recriação: {e}")
        conn.rollback()
        return False
    finally:
        conn.close()

    return verify_database()


def verify_database():
    """Verifica se o banco de dados e as tabelas foram criados corretamente."""
    print("\nPASSO 4: Verificando a estrutura do banco de dados...")

    load_dotenv()
    db_name = os.getenv('MYSQL_DB', 'qcyberDB')

    try:
        conn = pymysql.connect(
            host=os.getenv('MYSQL_HOST', 'localhost'),
            user=os.getenv('MYSQL_USER', 'root'),
            password=os.getenv('MYSQL_PASSWORD', 'root'),
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
        print("\n🎉 SUCESSO: BANCO DE DADOS RECRIADO COM SUCESSO!")
        return True
    except Exception as e:
        print(f"❌ ERRO na verificação: {e}")
        return False


def create_default_users():
    """Cria os usuários padrão: administrador e usuário de teste."""
    print("\nPASSO 5: Criando usuários padrão...")

    load_dotenv()
    db_name = os.getenv('MYSQL_DB', 'qcyberDB')

    try:
        conn = pymysql.connect(
            host=os.getenv('MYSQL_HOST', 'localhost'),
            user=os.getenv('MYSQL_USER', 'root'),
            password=os.getenv('MYSQL_PASSWORD', 'root'),
            database=db_name,
            charset='utf8mb4'
        )
        with conn.cursor() as cursor:
            # 1. Usuário Administrador
            admin_email = 'admin@qcyber.local'
            cursor.execute("SELECT id FROM usuarios WHERE email = %s", (admin_email,))
            if not cursor.fetchone():
                senha_hashed = hash_password('admin123456')
                cursor.execute("""
                               INSERT INTO usuarios (nome, email, senha_hash, is_admin, tem_permissao_sistema,
                                                     email_verificado)
                               VALUES (%s, %s, %s, %s, %s, %s)
                               """, ('Administrador do Sistema', admin_email, senha_hashed, True, True, True))
                print(f"✅ Usuário Administrador criado: {admin_email}")
            else:
                print(f"⚠️  Usuário '{admin_email}' já existe.")

            # 2. Usuário de Teste
            test_email = 'usuario@teste.com'
            cursor.execute("SELECT id FROM usuarios WHERE email = %s", (test_email,))
            if not cursor.fetchone():
                senha_hashed = hash_password('123456')
                cursor.execute("""
                               INSERT INTO usuarios (nome, email, senha_hash, is_admin, tem_permissao_sistema,
                                                     email_verificado, dois_fatores_ativo)
                               VALUES (%s, %s, %s, %s, %s, %s, %s)
                               """, ('Usuario de Teste', test_email, senha_hashed, False, True, True, False))
                print(f"✅ Usuário de Teste criado: {test_email}")
            else:
                print(f"⚠️  Usuário '{test_email}' já existe.")

        conn.commit()
        conn.close()

        print("\n--- Credenciais Padrão ---")
        print("Administrador:")
        print("  - Email: admin@qcyber.local")
        print("  - Senha: admin123456")
        print("Usuário de Teste:")
        print("  - Email: usuario@teste.com")
        print("  - Senha: 123456")
        return True

    except Exception as e:
        print(f"❌ ERRO ao criar usuários padrão: {e}")
        return False


if __name__ == '__main__':
    print("ATENÇÃO: Este script irá apagar e recriar o banco de dados 'qcyberDB'!")
    resposta = input("Você tem certeza que deseja continuar? (s/N): ")

    if resposta.lower() in ['s', 'sim', 'y', 'yes']:
        if recreate_database():
            create_default_users()
            print("\n🚀 Tudo pronto para começar!")
    else:
        print("Operação cancelada pelo usuário.")

