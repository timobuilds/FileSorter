from sqlalchemy import create_engine, Column, Integer, String, DateTime, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime

Base = declarative_base()

class Project(Base):
    __tablename__ = 'projects'
    
    id = Column(Integer, primary_key=True)
    name = Column(String, unique=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    files = relationship("File", back_populates="project")

class File(Base):
    __tablename__ = 'files'
    
    id = Column(Integer, primary_key=True)
    path = Column(String)
    original_name = Column(String)
    current_name = Column(String)
    file_type = Column(String)
    size = Column(Integer)
    summary = Column(String)
    project_id = Column(Integer, ForeignKey('projects.id'))
    tags = Column(String)  # Store as comma-separated values
    created_at = Column(DateTime, default=datetime.utcnow)
    project = relationship("Project", back_populates="files")

class DatabaseManager:
    def __init__(self, db_path="filesorter.db"):
        self.engine = create_engine(f'sqlite:///{db_path}')
        Base.metadata.create_all(self.engine)
        Session = sessionmaker(bind=self.engine)
        self.session = Session()
    
    def create_project(self, name):
        project = Project(name=name)
        self.session.add(project)
        self.session.commit()
        return project
    
    def add_file(self, project_id, file_data):
        file_entry = File(
            path=file_data['path'],
            original_name=file_data['original_name'],
            current_name=file_data['current_name'],
            file_type=file_data['file_type'],
            size=file_data['size'],
            summary=file_data.get('summary', ''),
            project_id=project_id,
            tags=','.join(file_data.get('tags', []))
        )
        self.session.add(file_entry)
        self.session.commit()
        return file_entry
    
    def get_projects(self):
        return self.session.query(Project).all()
    
    def get_files_by_project(self, project_id):
        return self.session.query(File).filter(File.project_id == project_id).all()
    
    def update_file_summary(self, file_id, summary):
        file_entry = self.session.query(File).filter(File.id == file_id).first()
        if file_entry:
            file_entry.summary = summary
            self.session.commit()
    
    def update_file_name(self, file_id, new_name):
        file_entry = self.session.query(File).filter(File.id == file_id).first()
        if file_entry:
            file_entry.current_name = new_name
            self.session.commit()
    
    def add_tags(self, file_id, tags):
        file_entry = self.session.query(File).filter(File.id == file_id).first()
        if file_entry:
            current_tags = set(file_entry.tags.split(',')) if file_entry.tags else set()
            current_tags.update(tags)
            file_entry.tags = ','.join(filter(None, current_tags))
            self.session.commit()

    def get_file_by_name(self, project_id, file_name):
        """Get file information by project ID and file name."""
        return self.session.query(File).filter(
            File.project_id == project_id,
            File.current_name == file_name
        ).first()

    def get_file_tags(self, file_id):
        """Get tags for a specific file."""
        file_entry = self.session.query(File).filter(File.id == file_id).first()
        if file_entry and file_entry.tags:
            return file_entry.tags.split(',')
        return []

    def remove_tags(self, file_id, tags_to_remove):
        """Remove specific tags from a file."""
        file_entry = self.session.query(File).filter(File.id == file_id).first()
        if file_entry and file_entry.tags:
            current_tags = set(file_entry.tags.split(','))
            current_tags.difference_update(tags_to_remove)
            file_entry.tags = ','.join(filter(None, current_tags))
            self.session.commit()
