import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import random
import io
import sqlite3
from sqlalchemy import create_engine
import requests
import base64
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

# Configuration de la page
st.set_page_config(
    page_title="Dashboard MDPH La Réunion - Version Avancée",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Styles CSS personnalisés
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        padding: 1rem;
        background: balck;
        border-radius: 10px;
        margin-bottom: 2rem;
        border-left: 5px solid #F97316;
        animation: fadeIn 1s;
    }
    .kpi-card {
        background-color: black;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        text-align: center;
        border-bottom: 3px solid #1E3A8A;
        transition: transform 0.3s;
    }
    .kpi-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.15);
    }
    .kpi-value {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E3A8A;
    }
    .kpi-label {
        font-size: 1rem;
        color: #666;
        margin-top: 0.5rem;
    }
    .alert-card {
        background-color: #FEF2F2;
        border-left: 4px solid #DC2626;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 5px;
        animation: slideIn 0.5s;
    }
    .success-card {
        background-color: #F0FDF4;
        border-left: 4px solid #16A34A;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 5px;
    }
    .info-card {
        background-color: #EFF6FF;
        border-left: 4px solid #2563EB;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 5px;
    }
    .team-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        text-align: center;
    }
    .stButton>button {
        background-color: #1E3A8A;
        color: white;
        border-radius: 5px;
        border: none;
        padding: 0.5rem 2rem;
        font-weight: bold;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #F97316;
        transform: scale(1.05);
    }
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    @keyframes slideIn {
        from { transform: translateX(-20px); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# 1. CONNEXION À LA BASE DE DONNÉES RÉELLE
# ============================================

class DatabaseConnector:
    """Gestionnaire de connexion à différentes bases de données"""
    
    def __init__(self):
        self.connection = None
        self.engine = None
        self.connection_string = None
        self.db_type = None
    
    def connect_sqlite(self, db_path="mdph_reunion.db"):
        """Connexion à une base SQLite locale"""
        try:
            self.connection = sqlite3.connect(db_path, check_same_thread=False)
            self.engine = create_engine(f'sqlite:///{db_path}')
            self.connection_string = f'sqlite:///{db_path}'
            self.db_type = 'sqlite'
            return True
        except Exception as e:
            st.error(f"❌ Erreur de connexion SQLite: {e}")
            return False
    
    def connect_postgresql(self, host, database, user, password, port=5432):
        """Connexion à PostgreSQL"""
        try:
            conn_string = f'postgresql://{user}:{password}@{host}:{port}/{database}'
            self.engine = create_engine(conn_string)
            self.connection = self.engine.connect()
            self.connection_string = conn_string
            self.db_type = 'postgresql'
            return True
        except Exception as e:
            st.error(f"❌ Erreur de connexion PostgreSQL: {e}")
            return False
    
    def execute_query(self, query):
        """Exécuter une requête SQL et retourner un DataFrame"""
        try:
            if self.engine:
                return pd.read_sql(query, self.engine)
            elif self.connection:
                return pd.read_sql_query(query, self.connection)
            else:
                return None
        except Exception as e:
            st.error(f"❌ Erreur d'exécution: {e}")
            return None
    
    def get_connection_info(self):
        """Retourne les infos de connexion pour le cache"""
        return {
            'db_type': self.db_type,
            'connection_string': self.connection_string
        }
    
    def close(self):
        """Fermer la connexion"""
        if self.connection:
            self.connection.close()

# ============================================
# 2. MODULE D'IA POUR LA PRÉDICTION DES DÉLAIS
# ============================================

class DelaiPredictor:
    """Modèle ML pour prédire les délais de traitement"""
    
    def __init__(self):
        self.model = None
        self.label_encoders = {}
        self.feature_columns = ['Type_Demande', 'Commune', 'Age_Groupe', 'Mois_Depot', 'Complet']
        self.is_trained = False
    
    def prepare_features(self, df):
        """Préparer les features pour le modèle"""
        df_features = df.copy()
        
        # Features temporelles
        df_features['Mois_Depot'] = pd.to_datetime(df_features['Date_Depot']).dt.month
        df_features['Annee_Depot'] = pd.to_datetime(df_features['Date_Depot']).dt.year
        df_features['Jour_Semaine'] = pd.to_datetime(df_features['Date_Depot']).dt.dayofweek
        
        # Encodage des variables catégorielles
        for col in ['Type_Demande', 'Commune', 'Age_Groupe']:
            if col in df_features.columns:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    df_features[col + '_encoded'] = self.label_encoders[col].fit_transform(df_features[col].astype(str))
                else:
                    # Gérer les nouvelles catégories
                    known_classes = set(self.label_encoders[col].classes_)
                    df_features[col] = df_features[col].apply(
                        lambda x: x if x in known_classes else 'Autre'
                    )
                    df_features[col + '_encoded'] = self.label_encoders[col].transform(df_features[col].astype(str))
        
        return df_features
    
    def train(self, df):
        """Entraîner le modèle sur les données historiques"""
        try:
            # Préparer les données
            df_train = df[df['Delai_Traitement_Jours'].notna()].copy()
            if len(df_train) < 100:
                return False, "⚠️ Données insuffisantes pour l'entraînement (<100 dossiers)"
            
            df_features = self.prepare_features(df_train)
            
            # Sélectionner les features
            feature_cols = [col + '_encoded' for col in ['Type_Demande', 'Commune', 'Age_Groupe']]
            feature_cols.extend(['Mois_Depot', 'Complet'])
            
            X = df_features[feature_cols].fillna(0)
            y = df_features['Delai_Traitement_Jours']
            
            # Split et entraînement
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            self.model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
            self.model.fit(X_train, y_train)
            
            # Évaluation
            score = self.model.score(X_test, y_test)
            self.is_trained = True
            
            return True, f"✅ Modèle entraîné avec succès (R² = {score:.3f})"
            
        except Exception as e:
            return False, f"❌ Erreur lors de l'entraînement: {e}"
    
    def predict(self, df):
        """Prédire les délais pour de nouveaux dossiers"""
        if not self.is_trained:
            return np.array([120] * len(df))
        
        try:
            df_features = self.prepare_features(df)
            feature_cols = [col + '_encoded' for col in ['Type_Demande', 'Commune', 'Age_Groupe']]
            feature_cols.extend(['Mois_Depot', 'Complet'])
            
            X_pred = df_features[feature_cols].fillna(0)
            predictions = self.model.predict(X_pred)
            
            return predictions
            
        except Exception as e:
            return np.array([120] * len(df))
    
    def get_feature_importance(self):
        """Obtenir l'importance des features"""
        if self.model and hasattr(self.model, 'feature_importances_'):
            features = ['Type', 'Commune', 'Âge', 'Mois', 'Complet']
            return features, self.model.feature_importances_
        return None, None
    
    def save_model(self, path="delai_predictor.pkl"):
        """Sauvegarder le modèle"""
        if self.is_trained:
            joblib.dump({
                'model': self.model,
                'label_encoders': self.label_encoders,
                'feature_columns': self.feature_columns
            }, path)
            return True
        return False
    
    def load_model(self, path="delai_predictor.pkl"):
        """Charger un modèle existant"""
        if os.path.exists(path):
            try:
                data = joblib.load(path)
                self.model = data['model']
                self.label_encoders = data['label_encoders']
                self.feature_columns = data['feature_columns']
                self.is_trained = True
                return True
            except:
                return False
        return False

# ============================================
# 3. GÉNÉRATION DE RAPPORTS PDF/EXCEL
# ============================================

class ReportGenerator:
    """Générateur de rapports PDF et Excel"""
    
    def __init__(self):
        self.styles = getSampleStyleSheet()
        self.setup_custom_styles()
    
    def setup_custom_styles(self):
        """Configuration des styles personnalisés"""
        self.styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#1E3A8A'),
            spaceAfter=30,
            alignment=1  # Center
        ))
        
        self.styles.add(ParagraphStyle(
            name='CustomHeading',
            parent=self.styles['Heading2'],
            fontSize=16,
            textColor=colors.HexColor('#F97316'),
            spaceAfter=20,
            spaceBefore=20
        ))
    
    def generate_excel_report(self, df, filename="rapport_mdph.xlsx"):
        """Générer un rapport Excel avec multiples onglets"""
        try:
            with pd.ExcelWriter(filename, engine='openpyxl') as writer:
                # Onglet principal
                df.to_excel(writer, sheet_name='Tous les dossiers', index=False)
                
                # Statistiques par type
                type_stats = df.groupby('Type_Demande').agg({
                    'ID_Dossier': 'count',
                    'Delai_Traitement_Jours': 'mean',
                    'Urgent': 'sum'
                }).round(1)
                type_stats.columns = ['Nombre', 'Délai moyen', 'Urgents']
                type_stats.to_excel(writer, sheet_name='Stats par type')
                
                # Statistiques par commune
                commune_stats = df.groupby('Commune').agg({
                    'ID_Dossier': 'count',
                    'Delai_Traitement_Jours': 'mean',
                    'Complet': 'mean'
                }).round(1)
                commune_stats.columns = ['Nombre', 'Délai moyen', 'Taux complétude']
                commune_stats.to_excel(writer, sheet_name='Stats par commune')
                
                # Alertes et urgences
                alertes = df[df['Alerte_Delai'] | df['Urgent']]
                alertes.to_excel(writer, sheet_name='Alertes', index=False)
                
                # Prévisions (si disponibles)
                if 'Delai_Predit' in df.columns:
                    prev = df[['ID_Dossier', 'Type_Demande', 'Delai_Traitement_Jours', 'Delai_Predit']].copy()
                    prev['Ecart'] = prev['Delai_Predit'] - prev['Delai_Traitement_Jours']
                    prev.to_excel(writer, sheet_name='Prévisions', index=False)
            
            return filename
            
        except Exception as e:
            st.error(f"❌ Erreur génération Excel: {e}")
            return None
    
    def generate_pdf_report(self, df, stats, filename="rapport_mdph.pdf"):
        """Générer un rapport PDF formaté"""
        try:
            doc = SimpleDocTemplate(filename, pagesize=A4)
            elements = []
            
            # Titre
            elements.append(Paragraph("Rapport MDPH La Réunion", self.styles['CustomTitle']))
            elements.append(Paragraph(f"Généré le {datetime.now().strftime('%d/%m/%Y %H:%M')}", self.styles['Normal']))
            elements.append(Spacer(1, 0.5*inch))
            
            # KPIs
            data_kpi = [
                ["Indicateur", "Valeur"],
                ["Total dossiers", str(stats['total'])],
                ["Délai moyen", f"{stats['delai_moyen']:.1f} jours"],
                ["Dossiers urgents", str(stats['urgents'])],
                ["Taux complétude", f"{stats['taux_completude']:.1f}%"]
            ]
            
            table_kpi = Table(data_kpi, colWidths=[200, 100])
            table_kpi.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1E3A8A')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 14),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            
            elements.append(Paragraph("Indicateurs clés", self.styles['CustomHeading']))
            elements.append(table_kpi)
            elements.append(Spacer(1, 0.5*inch))
            
            # Top 10 des communes
            top_communes = df['Commune'].value_counts().head(10)
            data_communes = [["Commune", "Nombre"]] + [[c, str(v)] for c, v in top_communes.items()]
            table_communes = Table(data_communes, colWidths=[200, 100])
            table_communes.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#F97316')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            
            elements.append(Paragraph("Top 10 des communes", self.styles['CustomHeading']))
            elements.append(table_communes)
            
            # Construction du PDF
            doc.build(elements)
            return filename
            
        except Exception as e:
            st.error(f"❌ Erreur génération PDF: {e}")
            return None
    
    def get_download_link(self, filename, text="Télécharger"):
        """Générer un lien de téléchargement"""
        with open(filename, 'rb') as f:
            data = f.read()
        b64 = base64.b64encode(data).decode()
        href = f'<a href="data:application/octet-stream;base64,{b64}" download="{filename}">{text}</a>'
        return href

# ============================================
# 4. INTÉGRATION D'API POUR NOTIFICATIONS
# ============================================

class NotificationService:
    """Service de notifications via API"""
    
    def __init__(self):
        self.webhook_url = None
    
    def configure_teams_webhook(self, url):
        """Configurer webhook Microsoft Teams"""
        self.webhook_url = url
    
    def send_teams_notification(self, message, title="Alerte MDPH"):
        """Envoyer une notification Teams"""
        if not self.webhook_url:
            return False
        
        try:
            card = {
                "@type": "MessageCard",
                "@context": "http://schema.org/extensions",
                "themeColor": "FF0000",
                "summary": title,
                "sections": [{
                    "activityTitle": title,
                    "activitySubtitle": "MDPH La Réunion",
                    "facts": [{
                        "name": "Message",
                        "value": message
                    }],
                    "markdown": True
                }]
            }
            
            response = requests.post(
                self.webhook_url,
                json=card,
                headers={'Content-Type': 'application/json'},
                timeout=5
            )
            
            return response.status_code == 200
            
        except Exception as e:
            st.error(f"❌ Erreur envoi Teams: {e}")
            return False
    
    def check_alerts_and_notify(self, df):
        """Vérifier les alertes et notifier"""
        alerts = []
        
        # Vérifier les délais dépassés
        delais_depasses = df[df['Alerte_Delai']]
        if len(delais_depasses) > 0:
            alerts.append(f"🔴 {len(delais_depasses)} dossiers ont dépassé le délai légal")
        
        # Vérifier les urgences
        urgents = df[df['Urgent']]
        if len(urgents) > 0:
            alerts.append(f"⚠️ {len(urgents)} dossiers urgents en attente")
        
        # Vérifier les échéances proches
        echeances = df[(df['Jours_Restants'] > 0) & (df['Jours_Restants'] <= 3)]
        if len(echeances) > 0:
            alerts.append(f"⏰ {len(echeances)} dossiers à échéance dans les 3 jours")
        
        # Envoyer les notifications
        if alerts and self.webhook_url:
            message = "\n\n".join(alerts)
            self.send_teams_notification(message, "🚨 Alertes MDPH - Action requise")
        
        return alerts

# ============================================
# 5. TABLEAU DE BORD PAR ÉQUIPE
# ============================================

class TeamDashboard:
    """Gestion des tableaux de bord par équipe"""
    
    def __init__(self):
        self.teams = {
            'ACCUEIL': {
                'name': 'Équipe Accueil',
                'color': '#3B82F6',
                'icon': '👋',
                'kpis': ['nouveaux_jour', 'appels_manques', 'satisfaction']
            },
            'INSTRUCTION': {
                'name': "Équipe d'Instruction",
                'color': '#F97316',
                'icon': '📋',
                'kpis': ['dossiers_instruits', 'delai_moyen', 'productivite']
            },
            'MEDICAL': {
                'name': 'Équipe Médicale',
                'color': '#10B981',
                'icon': '👨‍⚕️',
                'kpis': ['certificats_traites', 'en_attente_medical', 'duree_moyenne_expertise']
            },
            'COMMISSION': {
                'name': 'Commission',
                'color': '#8B5CF6',
                'icon': '⚖️',
                'kpis': ['dossiers_commission', 'prochaine_commission', 'en_attente_commission']
            },
            'ADMIN': {
                'name': 'Équipe Administrative',
                'color': '#6B7280',
                'icon': '📊',
                'kpis': ['courriers_envoyes', 'reclamations', 'traitement_post_decision']
            }
        }
    
    def get_team_stats(self, df, team):
        """Obtenir les statistiques pour une équipe spécifique"""
        stats = {}
        
        if team == 'ACCUEIL':
            stats['nouveaux_jour'] = len(df[df['Date_Depot'].dt.date == datetime.now().date()])
            stats['appels_manques'] = random.randint(0, 10)
            stats['satisfaction'] = random.randint(85, 98)
            
        elif team == 'INSTRUCTION':
            stats['dossiers_instruits'] = len(df[df['Statut'] == 'En instruction'])
            stats['delai_moyen'] = round(df['Delai_Traitement_Jours'].mean(), 1) if not pd.isna(df['Delai_Traitement_Jours'].mean()) else 0
            stats['productivite'] = random.randint(70, 95)
            
        elif team == 'MEDICAL':
            stats['certificats_traites'] = len(df[df['Statut'] == 'Complet'])
            stats['en_attente_medical'] = len(df[df['Statut'] == 'En attente pièces'])
            stats['duree_moyenne_expertise'] = random.randint(15, 30)
            
        elif team == 'COMMISSION':
            stats['dossiers_commission'] = len(df[df['Statut'] == 'Commission'])
            stats['prochaine_commission'] = (datetime.now() + timedelta(days=random.randint(1, 7))).strftime('%d/%m/%Y')
            stats['en_attente_commission'] = len(df[df['Statut'] == 'Commission'])
            
        elif team == 'ADMIN':
            stats['courriers_envoyes'] = random.randint(20, 50)
            stats['reclamations'] = random.randint(0, 5)
            stats['traitement_post_decision'] = random.randint(1, 5)
        
        return stats
    
    def display_team_dashboard(self, df):
        """Afficher le tableau de bord par équipe"""
        st.markdown("### 👥 Tableaux de bord par équipe")
        
        tabs = st.tabs([f"{info['icon']} {info['name']}" for info in self.teams.values()])
        
        for tab, (team_key, team_info) in zip(tabs, self.teams.items()):
            with tab:
                stats = self.get_team_stats(df, team_key)
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    for i, (kpi, value) in enumerate(list(stats.items())[:2]):
                        st.markdown(f"""
                        <div class="kpi-card" style="border-bottom-color: {team_info['color']}">
                            <div class="kpi-value" style="color: {team_info['color']}">{value}</div>
                            <div class="kpi-label">{kpi.replace('_', ' ').title()}</div>
                        </div>
                        """, unsafe_allow_html=True)
                        if i == 0:
                            st.markdown("<br>", unsafe_allow_html=True)
                
                with col2:
                    for i, (kpi, value) in enumerate(list(stats.items())[2:]):
                        st.markdown(f"""
                        <div class="kpi-card" style="border-bottom-color: {team_info['color']}">
                            <div class="kpi-value" style="color: {team_info['color']}">{value}</div>
                            <div class="kpi-label">{kpi.replace('_', ' ').title()}</div>
                        </div>
                        """, unsafe_allow_html=True)
                        if i == 0:
                            st.markdown("<br>", unsafe_allow_html=True)
                
                with col3:
                    st.markdown(f"""
                    <div class="team-card" style="background: linear-gradient(135deg, {team_info['color']} 0%, #764ba2 100%)">
                        <h3>{team_info['icon']} {team_info['name']}</h3>
                        <p>Actif • {datetime.now().strftime('%H:%M')}</p>
                        <progress value="75" max="100" style="width:100%; height:20px; border-radius:10px;"></progress>
                        <p style="margin-top:10px">Charge de travail: 75%</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Liste des tâches prioritaires
                    st.markdown("#### 📋 Tâches prioritaires")
                    tasks = {
                        'ACCUEIL': ["Traiter les nouveaux dossiers", "Répondre aux appels en attente", "Vérifier les documents reçus"],
                        'INSTRUCTION': ["Finaliser 5 dossiers urgents", "Mettre à jour les statuts", "Préparer réunion d'équipe"],
                        'MEDICAL': ["Examiner certificats en attente", "Rédiger avis pour commission", "Contacter médecins traitants"],
                        'COMMISSION': ["Préparer dossiers commission", "Vérifier disponibilité salle", "Envoyer convocations"],
                        'ADMIN': ["Traiter courriers sortants", "Mettre à jour base de données", "Préparer statistiques mensuelles"]
                    }
                    
                    for task in tasks[team_key][:3]:
                        st.markdown(f"- {task}")

# ============================================
# 6. FONCTION PRINCIPALE DE CHARGEMENT DES DONNÉES
# ============================================

@st.cache_data(ttl=3600)
def load_data_from_db(_db_connector=None, use_sample=True):
    """
    Charger les données depuis la DB ou utiliser un échantillon
    Le _ devant db_connector indique à Streamlit de ne pas hacher cet argument
    """
    
    if use_sample or _db_connector is None:
        # Données simulées
        return generate_sample_data()
    else:
        try:
            # Requête SQL réelle
            query = """
            SELECT 
                d.id_dossier as ID_Dossier,
                d.date_depot as Date_Depot,
                d.date_decision as Date_Decision,
                d.delai_traitement as Delai_Traitement_Jours,
                t.libelle as Type_Demande,
                s.libelle as Statut,
                c.nom as Commune,
                a.libelle as Age_Groupe,
                d.urgent as Urgent,
                d.complet as Complet,
                d.alerte_delai as Alerte_Delai,
                d.date_echeance as Date_Echeance,
                d.jours_restants as Jours_Restants
            FROM dossiers d
            LEFT JOIN type_demandes t ON d.type_demande_id = t.id
            LEFT JOIN statuts s ON d.statut_id = s.id
            LEFT JOIN communes c ON d.commune_id = c.id
            LEFT JOIN age_groupes a ON d.age_groupe_id = a.id
            WHERE d.date_depot >= '2023-01-01'
            ORDER BY d.date_depot DESC
            """
            
            df = _db_connector.execute_query(query)
            
            if df is not None and not df.empty:
                # Conversion des types
                df['Date_Depot'] = pd.to_datetime(df['Date_Depot'])
                df['Date_Decision'] = pd.to_datetime(df['Date_Decision'])
                df['Date_Echeance'] = pd.to_datetime(df['Date_Echeance'])
                
                return df
            else:
                st.warning("⚠️ Aucune donnée trouvée en base - Utilisation des données simulées")
                return generate_sample_data()
        except Exception as e:
            st.error(f"❌ Erreur lors du chargement des données: {e}")
            return generate_sample_data()

def generate_sample_data():
    """Générer des données d'exemple"""
    np.random.seed(42)
    
    # Types de demandes
    request_types = ['AAH', 'PCH', 'RQTH', 'Carte Mobilité Inclusion', 'AEEH', 'Prestations diverses']
    request_type_weights = [0.30, 0.25, 0.20, 0.15, 0.07, 0.03]
    
    # Statuts des dossiers
    statuses = ['Nouveau', 'En instruction', 'Complet', 'En attente pièces', 'Commission', 'Décision prise']
    status_weights = [0.10, 0.25, 0.15, 0.20, 0.15, 0.15]
    
    # Communes de La Réunion
    communes = [
        'Saint-Denis', 'Saint-Paul', 'Saint-Pierre', 'Le Tampon', 'Saint-André',
        'Saint-Louis', 'Saint-Benoît', 'Saint-Joseph', 'Saint-Leu', 'La Possession',
        'Sainte-Marie', 'Sainte-Suzanne', 'Bras-Panon', 'Salazie', 'Cilaos',
        'L\'Étang-Salé', 'Les Avirons', 'Saint-Philippe', 'Petite-Île', 'Trois-Bassins',
        'Sainte-Rose', 'La Plaine-des-Palmistes', 'Entre-Deux'
    ]
    
    # Profils d'âge
    age_groups = ['0-17 ans', '18-30 ans', '31-50 ans', '51-65 ans', '65+ ans']
    age_weights = [0.08, 0.22, 0.35, 0.25, 0.10]
    
    n_dossiers = 1000
    
    # Générer les dates de dépôt
    end_date = datetime.now()
    start_date = end_date - timedelta(days=730)
    deposit_dates = [start_date + timedelta(days=random.randint(0, 730)) for _ in range(n_dossiers)]
    
    # Calculer les dates de décision
    decision_dates = []
    delai_jours = []
    for date in deposit_dates:
        if random.random() < 0.7:
            delai = random.randint(30, 180)
            decision_date = date + timedelta(days=delai)
            if decision_date <= end_date:
                decision_dates.append(decision_date)
                delai_jours.append(delai)
            else:
                decision_dates.append(None)
                delai_jours.append(None)
        else:
            decision_dates.append(None)
            delai_jours.append(None)
    
    # Création du DataFrame
    df = pd.DataFrame({
        'ID_Dossier': [f'DOS-{2023000+i}' for i in range(n_dossiers)],
        'Date_Depot': deposit_dates,
        'Date_Decision': decision_dates,
        'Delai_Traitement_Jours': delai_jours,
        'Type_Demande': np.random.choice(request_types, n_dossiers, p=request_type_weights),
        'Statut': np.random.choice(statuses, n_dossiers, p=status_weights),
        'Commune': np.random.choice(communes, n_dossiers),
        'Age_Groupe': np.random.choice(age_groups, n_dossiers, p=age_weights),
        'Urgent': np.random.choice([True, False], n_dossiers, p=[0.15, 0.85]),
        'Complet': np.random.choice([True, False], n_dossiers, p=[0.70, 0.30]),
    })
    
    # Calculer les alertes de délai
    df['Alerte_Delai'] = df.apply(
        lambda row: row['Delai_Traitement_Jours'] is not None and row['Delai_Traitement_Jours'] > 120, 
        axis=1
    )
    
    # Ajouter des échéances
    df['Date_Echeance'] = df['Date_Depot'] + timedelta(days=120)
    df['Jours_Restants'] = df.apply(
        lambda row: (row['Date_Echeance'] - datetime.now()).days 
        if pd.isna(row['Date_Decision']) and row['Date_Echeance'] > datetime.now() 
        else -1, axis=1
    )
    
    return df

# ============================================
# 7. INTERFACE PRINCIPALE STREAMLIT
# ============================================

def main():
    # En-tête
    st.markdown('<h1 class="main-header">🏥 Tableau de Bord MDPH - La Réunion (Version Avancée)</h1>', unsafe_allow_html=True)
    
    # Initialisation des services dans session_state
    if 'db_connector' not in st.session_state:
        st.session_state.db_connector = DatabaseConnector()
    if 'predictor' not in st.session_state:
        st.session_state.predictor = DelaiPredictor()
    if 'notifier' not in st.session_state:
        st.session_state.notifier = NotificationService()
    if 'reporter' not in st.session_state:
        st.session_state.reporter = ReportGenerator()
    if 'team_dashboard' not in st.session_state:
        st.session_state.team_dashboard = TeamDashboard()
    if 'use_sample' not in st.session_state:
        st.session_state.use_sample = True
    
    # Sidebar - Configuration avancée
    with st.sidebar:
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/b/b1/Logo_MDPH.svg/1200px-Logo_MDPH.svg.png", width=200)
        
        st.markdown("## ⚙️ Configuration")
        
        # Onglets de configuration
        config_tab1, config_tab2, config_tab3, config_tab4 = st.tabs(["📊 Données", "🤖 IA", "🔔 Notifications", "📤 Export"])
        
        with config_tab1:
            st.markdown("### Connexion Base de Données")
            
            db_type = st.radio("Type de base", ["SQLite (local)", "PostgreSQL", "Données simulées"])
            
            if db_type == "SQLite (local)":
                db_path = st.text_input("Chemin du fichier SQLite", "mdph_reunion.db")
                if st.button("🔄 Connecter SQLite"):
                    if st.session_state.db_connector.connect_sqlite(db_path):
                        st.session_state.use_sample = False
                        st.rerun()
            
            elif db_type == "PostgreSQL":
                with st.form("postgres_form"):
                    host = st.text_input("Host", "localhost")
                    port = st.number_input("Port", 5432)
                    database = st.text_input("Database", "mdph_reunion")
                    user = st.text_input("User", "postgres")
                    password = st.text_input("Password", type="password")
                    
                    if st.form_submit_button("🔌 Connecter PostgreSQL"):
                        if st.session_state.db_connector.connect_postgresql(host, database, user, password, port):
                            st.session_state.use_sample = False
                            st.rerun()
            
            else:
                st.session_state.use_sample = True
                st.info("📊 Utilisation des données simulées")
        
        with config_tab2:
            st.markdown("### Module de Prédiction IA")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🎯 Entraîner modèle"):
                    with st.spinner("Entraînement en cours..."):
                        # Charger les données pour l'entraînement
                        train_df = load_data_from_db(
                            st.session_state.db_connector if not st.session_state.use_sample else None,
                            use_sample=st.session_state.use_sample
                        )
                        success, message = st.session_state.predictor.train(train_df)
                        if success:
                            st.session_state.predictor.save_model()
                            st.success(message)
                        else:
                            st.error(message)
            
            with col2:
                if st.button("📂 Charger modèle"):
                    if st.session_state.predictor.load_model():
                        st.success("✅ Modèle chargé")
                    else:
                        st.warning("⚠️ Aucun modèle trouvé")
            
            # Importance des features
            if st.session_state.predictor.is_trained:
                features, importance = st.session_state.predictor.get_feature_importance()
                if importance is not None:
                    st.markdown("#### Importance des features")
                    fig = px.bar(x=features, y=importance, title="Impact sur les délais")
                    st.plotly_chart(fig, use_container_width=True)
        
        with config_tab3:
            st.markdown("### Notifications Automatiques")
            
            teams_url = st.text_input("Webhook Microsoft Teams")
            if st.button("✅ Configurer Teams"):
                st.session_state.notifier.configure_teams_webhook(teams_url)
                st.success("✅ Webhook configuré")
            
            if st.button("🔔 Tester notification"):
                test_msg = "Test notification - " + datetime.now().strftime("%H:%M:%S")
                if st.session_state.notifier.send_teams_notification(test_msg, "🧪 Test MDPH"):
                    st.success("✅ Notification envoyée")
                else:
                    st.error("❌ Échec envoi (vérifiez l'URL)")
        
        with config_tab4:
            st.markdown("### Export de rapports")
            
            if st.button("📊 Générer rapport Excel"):
                with st.spinner("Génération en cours..."):
                    # Charger les données
                    export_df = load_data_from_db(
                        st.session_state.db_connector if not st.session_state.use_sample else None,
                        use_sample=st.session_state.use_sample
                    )
                    
                    # Calcul des stats
                    stats = {
                        'total': len(export_df),
                        'delai_moyen': export_df['Delai_Traitement_Jours'].mean() if not pd.isna(export_df['Delai_Traitement_Jours'].mean()) else 0,
                        'urgents': export_df['Urgent'].sum(),
                        'taux_completude': export_df['Complet'].mean() * 100 if not pd.isna(export_df['Complet'].mean()) else 0
                    }
                    
                    filename = f"rapport_mdph_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx"
                    excel_file = st.session_state.reporter.generate_excel_report(export_df, filename)
                    
                    if excel_file:
                        href = st.session_state.reporter.get_download_link(excel_file, "📥 Télécharger Excel")
                        st.markdown(href, unsafe_allow_html=True)
            
            if st.button("📄 Générer rapport PDF"):
                with st.spinner("Génération du PDF..."):
                    # Charger les données
                    export_df = load_data_from_db(
                        st.session_state.db_connector if not st.session_state.use_sample else None,
                        use_sample=st.session_state.use_sample
                    )
                    
                    stats = {
                        'total': len(export_df),
                        'delai_moyen': export_df['Delai_Traitement_Jours'].mean() if not pd.isna(export_df['Delai_Traitement_Jours'].mean()) else 0,
                        'urgents': export_df['Urgent'].sum(),
                        'taux_completude': export_df['Complet'].mean() * 100 if not pd.isna(export_df['Complet'].mean()) else 0
                    }
                    
                    filename = f"rapport_mdph_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf"
                    pdf_file = st.session_state.reporter.generate_pdf_report(export_df, stats, filename)
                    
                    if pdf_file:
                        href = st.session_state.reporter.get_download_link(pdf_file, "📥 Télécharger PDF")
                        st.markdown(href, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Filtres standards
        st.markdown("## 🔍 Filtres")
        
        # Chargement des données avec le nouveau paramètre underscore
        df = load_data_from_db(
            _db_connector=st.session_state.db_connector if not st.session_state.use_sample else None,
            use_sample=st.session_state.use_sample
        )
        
        # Application des filtres
        selected_types = st.multiselect(
            "Types de demande",
            options=df['Type_Demande'].unique(),
            default=df['Type_Demande'].unique()
        )
        
        selected_status = st.multiselect(
            "Statuts",
            options=df['Statut'].unique(),
            default=df['Statut'].unique()
        )
        
        selected_communes = st.multiselect(
            "Communes",
            options=df['Commune'].unique(),
            default=df['Commune'].unique()[:5]
        )
        
        show_urgent_only = st.checkbox("⚠️ Afficher uniquement les urgents")
        
        # Application des filtres
        filtered_df = df[
            (df['Type_Demande'].isin(selected_types)) &
            (df['Statut'].isin(selected_status)) &
            (df['Commune'].isin(selected_communes))
        ]
        
        if show_urgent_only:
            filtered_df = filtered_df[filtered_df['Urgent']]
    
    # ============================================
    # CORPS PRINCIPAL DU DASHBOARD
    # ============================================
    
    # Ajout des prédictions si le modèle est entraîné
    if st.session_state.predictor.is_trained:
        with st.spinner("Calcul des prédictions..."):
            predictions = st.session_state.predictor.predict(filtered_df)
            filtered_df['Delai_Predit'] = predictions
            filtered_df['Ecart_Prediction'] = filtered_df['Delai_Predit'] - filtered_df['Delai_Traitement_Jours'].fillna(filtered_df['Delai_Predit'])
    
    # KPIs principaux
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        total_dossiers = len(filtered_df)
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-value">{total_dossiers}</div>
            <div class="kpi-label">📋 Dossiers actifs</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        delai_moyen = filtered_df['Delai_Traitement_Jours'].mean()
        delai_moyen = round(delai_moyen, 1) if not pd.isna(delai_moyen) else 0
        color = "#16A34A" if delai_moyen < 120 else "#DC2626"
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-value" style="color: {color};">{delai_moyen} jours</div>
            <div class="kpi-label">⏱️ Délai moyen réel</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        if 'Delai_Predit' in filtered_df.columns:
            delai_pred = filtered_df['Delai_Predit'].mean()
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-value" style="color: #8B5CF6;">{delai_pred:.1f} jours</div>
                <div class="kpi-label">🤖 Délai prédit (IA)</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            dossiers_urgents = filtered_df['Urgent'].sum()
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-value" style="color: #DC2626;">{dossiers_urgents}</div>
                <div class="kpi-label">⚠️ Dossiers urgents</div>
            </div>
            """, unsafe_allow_html=True)
    
    with col4:
        if 'Delai_Predit' in filtered_df.columns:
            dossiers_urgents = filtered_df['Urgent'].sum()
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-value" style="color: #DC2626;">{dossiers_urgents}</div>
                <div class="kpi-label">⚠️ Dossiers urgents</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            dossiers_incomplets = (~filtered_df['Complet']).sum()
            taux_incomplet = (dossiers_incomplets / total_dossiers * 100) if total_dossiers > 0 else 0
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-value">{dossiers_incomplets}</div>
                <div class="kpi-label">📄 Dossiers incomplets ({taux_incomplet:.1f}%)</div>
            </div>
            """, unsafe_allow_html=True)
    
    with col5:
        delai_depasse = filtered_df['Alerte_Delai'].sum()
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-value" style="color: #DC2626;">{delai_depasse}</div>
            <div class="kpi-label">⏰ Délais légaux dépassés</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Onglets principaux
    main_tab1, main_tab2, main_tab3, main_tab4, main_tab5 = st.tabs([
        "📊 Vue d'ensemble",
        "👥 Par équipe",
        "🤖 Analyses IA",
        "🚨 Alertes",
        "📋 Suivi détaillé"
    ])
    
    with main_tab1:
        # Graphiques principaux
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📊 Répartition par type de demande")
            type_counts = filtered_df['Type_Demande'].value_counts().reset_index()
            type_counts.columns = ['Type', 'Nombre']
            
            fig = px.pie(type_counts, values='Nombre', names='Type', 
                        color_discrete_sequence=px.colors.sequential.Blues_r,
                        hole=0.3)
            fig.update_traces(textposition='inside', textinfo='percent+label')
            fig.update_layout(showlegend=False, height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### ⏱️ Évolution des délais")
            
            df_time = filtered_df[filtered_df['Delai_Traitement_Jours'].notna()].copy()
            if not df_time.empty:
                df_time['Mois'] = df_time['Date_Depot'].dt.to_period('M').astype(str)
                monthly_delai = df_time.groupby('Mois')['Delai_Traitement_Jours'].mean().reset_index()
                
                fig = px.line(monthly_delai, x='Mois', y='Delai_Traitement_Jours',
                             markers=True)
                fig.add_hline(y=120, line_dash="dash", line_color="red", 
                             annotation_text="Délai légal (4 mois)")
                
                if 'Delai_Predit' in filtered_df.columns:
                    monthly_pred = df_time.groupby('Mois')['Delai_Predit'].mean().reset_index()
                    fig.add_scatter(x=monthly_pred['Mois'], y=monthly_pred['Delai_Predit'],
                                   mode='lines+markers', name='Prédiction IA',
                                   line=dict(color='purple', dash='dot'))
                
                fig.update_layout(height=400, xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Aucune donnée disponible pour l'évolution des délais")
        
        # Deuxième ligne
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📈 Statut des dossiers")
            status_counts = filtered_df['Statut'].value_counts().reset_index()
            status_counts.columns = ['Statut', 'Nombre']
            
            colors_map = {'Nouveau': '#3B82F6', 'En instruction': '#F59E0B', 
                         'Complet': '#10B981', 'En attente pièces': '#EF4444',
                         'Commission': '#8B5CF6', 'Décision prise': '#6B7280'}
            
            fig = px.bar(status_counts, x='Statut', y='Nombre', 
                        color='Statut', color_discrete_map=colors_map)
            fig.update_layout(showlegend=False, height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### 🗺️ Top 10 communes")
            geo_counts = filtered_df['Commune'].value_counts().head(10).reset_index()
            geo_counts.columns = ['Commune', 'Nombre']
            
            fig = px.bar(geo_counts, x='Nombre', y='Commune', orientation='h',
                        color='Nombre', color_continuous_scale='Blues')
            fig.update_layout(height=400, xaxis_title="Nombre de dossiers", yaxis_title="")
            st.plotly_chart(fig, use_container_width=True)
    
    with main_tab2:
        st.session_state.team_dashboard.display_team_dashboard(filtered_df)
    
    with main_tab3:
        st.markdown("### 🤖 Analyses prédictives IA")
        
        if 'Delai_Predit' in filtered_df.columns:
            col1, col2 = st.columns(2)
            
            with col1:
                # Comparaison réel vs prédit
                comparison = filtered_df[filtered_df['Delai_Traitement_Jours'].notna()].copy()
                if not comparison.empty:
                    comparison = comparison.head(20)  # Top 20 pour la lisibilité
                    
                    fig = go.Figure()
                    fig.add_trace(go.Bar(name='Réel', x=comparison['ID_Dossier'], 
                                         y=comparison['Delai_Traitement_Jours']))
                    fig.add_trace(go.Bar(name='Prédit IA', x=comparison['ID_Dossier'], 
                                         y=comparison['Delai_Predit']))
                    fig.update_layout(barmode='group', title="Comparaison Réel vs Prédit (20 derniers)",
                                     xaxis_tickangle=-45, height=400)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Aucun dossier traité pour la comparaison")
            
            with col2:
                # Distribution des écarts
                ecarts = filtered_df['Ecart_Prediction'].dropna()
                if len(ecarts) > 0:
                    fig = px.histogram(ecarts, nbins=30, title="Distribution des écarts de prédiction",
                                      labels={'value': 'Écart (jours)', 'count': 'Fréquence'},
                                      color_discrete_sequence=['purple'])
                    fig.add_vline(x=0, line_dash="dash", line_color="red")
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Aucune donnée pour la distribution des écarts")
            
            # Dossiers à risque selon l'IA
            st.markdown("### ⚠️ Dossiers à risque (selon IA)")
            
            at_risk = filtered_df[
                (filtered_df['Delai_Predit'] > 120) & 
                (filtered_df['Date_Decision'].isna())
            ].sort_values('Delai_Predit', ascending=False).head(10)
            
            if len(at_risk) > 0:
                risk_display = at_risk[['ID_Dossier', 'Type_Demande', 'Commune', 
                                        'Delai_Predit', 'Jours_Restants']].copy()
                risk_display['Delai_Predit'] = risk_display['Delai_Predit'].round(1)
                st.dataframe(risk_display, use_container_width=True, hide_index=True)
            else:
                st.success("✅ Aucun dossier à risque identifié")
        
        else:
            st.info("🤖 Entraînez le modèle IA dans la sidebar pour voir les prédictions")
    
    with main_tab4:
        st.markdown("### 🚨 Alertes et notifications")
        
        # Vérifier et afficher les alertes
        alerts = st.session_state.notifier.check_alerts_and_notify(filtered_df)
        
        if alerts:
            for alert in alerts:
                st.markdown(f'<div class="alert-card">{alert}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="success-card">✅ Aucune alerte active</div>', unsafe_allow_html=True)
        
        # Dossiers en dépassement
        st.markdown("#### ⏰ Dossiers en dépassement de délai")
        delais_depasses = filtered_df[filtered_df['Alerte_Delai']].head(10)
        
        if len(delais_depasses) > 0:
            display_delais = delais_depasses[['ID_Dossier', 'Type_Demande', 'Commune', 
                                              'Delai_Traitement_Jours', 'Date_Depot']].copy()
            display_delais['Date_Depot'] = display_delais['Date_Depot'].dt.strftime('%d/%m/%Y')
            st.dataframe(display_delais, use_container_width=True, hide_index=True)
        else:
            st.info("Aucun dossier en dépassement")
        
        # Échéances à venir
        st.markdown("#### 📅 Échéances des 7 prochains jours")
        echeances = filtered_df[
            (filtered_df['Jours_Restants'] > 0) & 
            (filtered_df['Jours_Restants'] <= 7)
        ].head(10)
        
        if len(echeances) > 0:
            display_echeances = echeances[['ID_Dossier', 'Type_Demande', 'Commune', 
                                          'Jours_Restants', 'Date_Echeance']].copy()
            display_echeances['Date_Echeance'] = display_echeances['Date_Echeance'].dt.strftime('%d/%m/%Y')
            st.dataframe(display_echeances, use_container_width=True, hide_index=True)
        else:
            st.info("Aucune échéance dans les 7 jours")
    
    with main_tab5:
        st.markdown("### 📋 Suivi détaillé des dossiers")
        
        # Recherche et filtres avancés
        search_id = st.text_input("🔍 Rechercher par ID dossier")
        
        if search_id:
            search_df = filtered_df[filtered_df['ID_Dossier'].str.contains(search_id, case=False, na=False)]
        else:
            search_df = filtered_df
        
        # Tableau avec mise en forme conditionnelle
        display_cols = ['ID_Dossier', 'Date_Depot', 'Type_Demande', 'Statut', 
                       'Commune', 'Urgent', 'Complet', 'Jours_Restants']
        
        if 'Delai_Predit' in search_df.columns:
            display_cols.append('Delai_Predit')
        
        df_display = search_df[display_cols].copy()
        df_display['Date_Depot'] = df_display['Date_Depot'].dt.strftime('%d/%m/%Y')
        df_display['Urgent'] = df_display['Urgent'].map({True: '🔴 Oui', False: '🟢 Non'})
        df_display['Complet'] = df_display['Complet'].map({True: '✅ Oui', False: '❌ Non'})
        
        # Configuration des colonnes avec couleurs
        column_config = {
            "Jours_Restants": st.column_config.NumberColumn(
                "Jours restants",
                help="Jours avant échéance légale",
                format="%d jours"
            )
        }
        
        if 'Delai_Predit' in df_display.columns:
            column_config["Delai_Predit"] = st.column_config.NumberColumn(
                "Délai prédit (IA)",
                help="Prédiction IA du délai de traitement",
                format="%.1f jours"
            )
        
        st.dataframe(
            df_display,
            use_container_width=True,
            hide_index=True,
            column_config=column_config
        )
        
        # Export de la vue actuelle
        if st.button("📥 Exporter cette vue en CSV"):
            csv = df_display.to_csv(index=False).encode('utf-8')
            b64 = base64.b64encode(csv).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="export_mdph.csv">Télécharger CSV</a>'
            st.markdown(href, unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    db_status = "Base de données réelle" if not st.session_state.get('use_sample', True) else "Données simulées"
    ia_status = "Entraîné" if st.session_state.predictor.is_trained else "Non entraîné"
    
    st.markdown(f"""
    <div style="text-align: center; color: #666; padding: 1rem;">
        <p>🏥 MDPH de La Réunion - Tableau de bord avancé v2.0</p>
        <p>⏱️ Dernière mise à jour: {datetime.now().strftime("%d/%m/%Y %H:%M")}</p>
        <p>🔌 Connecté à: {db_status}</p>
        <p>🤖 Modèle IA: {ia_status}</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
