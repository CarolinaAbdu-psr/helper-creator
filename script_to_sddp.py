import psr.factory 
import os

class ScriptInterpreter:
    def __init__(self,study):
        """
        Inicializa o interpretador com o objeto study da API
        
        Args:
            study: Objeto da API que contém os métodos create(), save(), etc.
        """
        self.study = study
        self.current_object = None
        self.objects_registry = {}  # Armazena objetos criados por nome
        
    def parse_and_execute(self, script):
        """
        Processa o script completo linha por linha
        
        Args:
            script: String contendo o script completo
        """
        lines = script.strip().split('\n')
        print(lines)
        
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):  # Ignora linhas vazias e comentários
                continue
            print(f"Executing line {line}")
            self._execute_line(line)
    
    def _execute_line(self, line):
        """Executa uma linha individual do script"""
        parts = line.split(None, 1)  # Divide no primeiro espaço
        if not parts:
            return
        
        command = parts[0].upper()
        args = parts[1] if len(parts) > 1 else ""
        
        if command == "CREATE_STUDY":
            self._create_study()
        elif command == "CREATE":
            self._create_object(args)
        elif command == "NAME":
            self._set_name(args)
        elif command == "CODE":
            self._set_code(args)
        elif command == "ID":
            self._set_id(args)
        elif command == "SET":
            self._set_property(args)
        elif command == "REF":
            self._set_reference(args)
        elif command == "ADD":
            self._add_object(args)
        elif command == "SAVE_STUDY":
            self._save_study()
        else:
            print(f"Comando desconhecido: {command}")
    
    def _create_study(self):
        """Executa CREATE_STUDY"""
        self.study=psr.factory.create_study()
        for obj in self.study.get_all_objects():
            self.study.remove(obj)
        print("Criando estudo...")
       
    
    def _create_object(self, object_type):
        """
        Executa CREATE <ObjectType>
        Exemplo: CREATE System -> study.create("System")
        """
        self.current_object = self.study.create(str(object_type))
        print(f"Objeto criado: {object_type}")
    
    def _set_name(self,name):
        """
        Add name to the object if required 
        """
        self.current_object.name = str(name)

    def _set_code(self,code):
        """
        Add code to the object if required 
        """
        self.current_object.code = int(code)

    def _set_id(self,id):
        """
        Add id to the object if required 
        """
        self.current_object.id = str(id)

    def _set_property(self, args):
        """
        Executa SET property : value
        Exemplo: SET name : System1 -> current_object.set("name", "System1")
        """
        if not self.current_object:
            print("Erro: Nenhum objeto atual para SET")
            return
        
        # Parse "property : value"
        parts = args.split(':', 1)
        if len(parts) != 2:
            print(f"Erro: Formato inválido para SET: {args}")
            return
        
        prop_name = parts[0].strip()
        prop_value = parts[1].strip()
        
        self.current_object.set(prop_name, prop_value)
        print(f"  SET {prop_name} = {prop_value}")
        
    
    def _set_reference(self, args):
        """
        Executa REF RefProperty : ReferencedObjectName
        Exemplo: REF RefSystem : System1 -> current_object.set("RefSystem", system1_object)
        """
        if not self.current_object:
            print("Erro: Nenhum objeto atual para REF")
            return
        
        # Parse "RefProperty : ObjectName"
        parts = args.split(':', 1)
        if len(parts) != 2:
            print(f"Erro: Formato inválido para REF: {args}")
            return
        
        ref_name = parts[0].strip()
        target_name = parts[1].strip()
        
        # Busca o objeto referenciado no registro
        if target_name in self.objects_registry.keys():
            referenced_object = self.study.find_by_name(self.objects_registry[target_name],target_name)[0]
            self.current_object.set(ref_name, referenced_object)
            print(f"  REF {ref_name} -> {target_name}")
        else:
            print(f"Erro: Objeto '{target_name}' não encontrado no registro")
    
    def _add_object(self, object_type):
        """
        Executa ADD <ObjectType>
        Finaliza a criação do objeto atual
        """
        if not self.current_object:
            print("Erro: Nenhum objeto atual para ADD")
            return
        
        self.study.add(self.current_object)
        print(f"Objeto {object_type} adicionado ao estudo")

        self.objects_registry[self.current_object.name]=object_type

        self.current_object = None  # Limpa o objeto atual

    def _save_study(self):
        """Executa SAVE_STUDY"""
        path = "CaseExample"
        os.makedirs(exist_ok=path)
        self.study.save(path)
        print("Estudo salvo com sucesso!")



# Exemplo de uso
if __name__ == "__main__":
    
    with open('script.txt', 'r', encoding='utf-8') as script_txt:
        script = script_txt.read()

    study = None

    print(script)
    
    # Execute o script
    interpreter = ScriptInterpreter(study)
    interpreter.parse_and_execute(script)