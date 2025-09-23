# app/models.py
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
from sqlalchemy import func

db = SQLAlchemy()

class Account(db.Model):
    """Model for storing account information"""
    __tablename__ = 'accounts'
    
    id = db.Column(db.Integer, primary_key=True)
    account_number = db.Column(db.String(50), nullable=False, index=True)
    disco = db.Column(db.String(20), nullable=False, index=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationship to bills
    bills = db.relationship('BillData', backref='account_ref', lazy='dynamic')
    
    # Unique constraint on account_number + disco combination
    __table_args__ = (db.UniqueConstraint('account_number', 'disco', name='unique_account_disco'),)
    
    def __repr__(self):
        return f'<Account {self.disco}-{self.account_number}>'


class BillData(db.Model):
    """Model for storing extracted bill data"""
    __tablename__ = 'bill_data'
    
    id = db.Column(db.Integer, primary_key=True)
    account_id = db.Column(db.Integer, db.ForeignKey('accounts.id'), nullable=False, index=True)
    
    # Basic bill information
    reference_no = db.Column(db.String(100), index=True)
    bill_month = db.Column(db.String(20), index=True)
    tariff = db.Column(db.String(50))
    
    # Load information
    sanctioned_load = db.Column(db.Float)
    connected_load = db.Column(db.Float)
    
    # K-Electric specific fields (for K-Electric bills)
    active_units_on_peak = db.Column(db.Float)
    active_units_off_peak = db.Column(db.Float)
    reactive_units_on_peak = db.Column(db.Float)
    reactive_units_off_peak = db.Column(db.Float)
    mdi_on_peak = db.Column(db.Float)
    mdi_off_peak = db.Column(db.Float)
    on_peak_unit_rate_old = db.Column(db.Float)
    on_peak_unit_rate_new = db.Column(db.Float)
    off_peak_unit_rate_old = db.Column(db.Float)
    off_peak_unit_rate_new = db.Column(db.Float)
    
    # General DISCO fields (for other DISCOs)
    kwh_units_consumed_peak = db.Column(db.Float)
    kwh_units_consumed_off_peak = db.Column(db.Float)
    kvarh_reading_peak = db.Column(db.Float)
    kvarh_reading_off_peak = db.Column(db.Float)
    mdi_reading_peak = db.Column(db.Float)
    mdi_reading_off_peak = db.Column(db.Float)
    off_peak_unit_rate = db.Column(db.Float)
    on_peak_unit_rate = db.Column(db.Float)
    
    # Financial information
    bill_amount = db.Column(db.Float)
    payable_within_due_date = db.Column(db.Float)
    lpf_penalty = db.Column(db.Float)
    pf_penalty = db.Column(db.Float)
    
    # Carbon footprint calculations
    total_units_consumed = db.Column(db.Float)
    co2_tonnes = db.Column(db.Float)
    ch4_tonnes = db.Column(db.Float)
    n2o_tonnes = db.Column(db.Float)
    net_carbon_footprint_tonnes = db.Column(db.Float)
    
    # Metadata
    file_name = db.Column(db.String(255))
    file_path = db.Column(db.String(500))
    processing_status = db.Column(db.String(20), default='SUCCESS')
    error_message = db.Column(db.Text)
    
    # Timestamps
    created_at = db.Column(db.DateTime, default=datetime.utcnow, index=True)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def __repr__(self):
        return f'<BillData {self.reference_no} - {self.bill_month}>'


class ProcessingLog(db.Model):
    """Model for storing processing logs and statistics"""
    __tablename__ = 'processing_logs'
    
    id = db.Column(db.Integer, primary_key=True)
    disco = db.Column(db.String(20), nullable=False, index=True)
    process_type = db.Column(db.String(20), nullable=False)  # 'single', 'all'
    total_accounts = db.Column(db.Integer, default=0)
    successful_extractions = db.Column(db.Integer, default=0)
    failed_extractions = db.Column(db.Integer, default=0)
    success_rate = db.Column(db.Float)
    total_carbon_footprint = db.Column(db.Float)
    
    # Timestamps
    started_at = db.Column(db.DateTime, default=datetime.utcnow)
    completed_at = db.Column(db.DateTime)
    
    def __repr__(self):
        return f'<ProcessingLog {self.disco} - {self.started_at}>'
