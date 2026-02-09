from dataclasses import dataclass, asdict
from typing import Dict, Any


@dataclass
class ClueStats:
    len: int                     
    sim: float                          
    novel: float                            
    risk: str                                       
    ts: int                      
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ClueStats':
        return cls(
            len=data.get('len', 0),
            sim=data.get('sim', 0.0),
            novel=data.get('novel', 1.0),
            risk=data.get('risk', 'low'),
            ts=data.get('ts', 0)
        )


@dataclass
class Clue:
    id: str                                   
    type: str                                                   
    peek: str                                  
    ptr: str                                                         
    stats: ClueStats                 
    content: str = None                           
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'type': self.type,
            'peek': self.peek,
            'ptr': self.ptr,
            'stats': self.stats.to_dict(),
            'content': self.content
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Clue':
        stats_data = data.get('stats', {})
        if isinstance(stats_data, ClueStats):
            stats = stats_data
        else:
            stats = ClueStats.from_dict(stats_data)
        
        return cls(
            id=data['id'],
            type=data['type'],
            peek=data['peek'],
            ptr=data['ptr'],
            stats=stats,
            content=data.get('content')
        )
    
    def __repr__(self) -> str:
        return f"Clue(id={self.id}, type={self.type}, sim={self.stats.sim:.2f})"

