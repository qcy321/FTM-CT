import ast
from abc import ABC, abstractmethod


def generate_func(node: ast.FunctionDef):
    """
    生成函数信息对象
    :param node: ast的函数节点信息
    :return:
    """
    return FunctionNode(node)


def generate_class(node: ast.ClassDef):
    """
    生成类信息对象
    :param node: ast的类节点信息
    :return:
    """
    body = ClassNode(node)
    for inner_node in node.body:
        if isinstance(inner_node, ast.FunctionDef):
            body.functions.append(generate_func(inner_node))
        elif isinstance(inner_node, ast.ClassDef):
            body.inner_classes.append(generate_class(inner_node))
    return body


class CommentRemover(ast.NodeTransformer):
    def visit_FunctionDef(self, node):
        # 过滤掉函数体中所有的 Expr 节点，如果它们是 Str 类型（即多行注释或 docstring）
        node.body = [
            n for n in node.body
            if not (isinstance(n, ast.Expr) and isinstance(n.value, ast.Str))
        ]
        return self.generic_visit(node)


class FunctionNode:
    """
    函数信息保存对象
    """

    def __init__(self, node) -> None:
        docstring = ast.get_docstring(node)
        remover = CommentRemover()
        source_code = ast.unparse(remover.visit(node))
        lines = [line for line in source_code.splitlines() if line.strip()]
        self.name = node.name if node.name else ""
        self.docstring = docstring if docstring else ""
        self.source_code = "\n".join(lines)


class ClassNode:
    """
    类信息保存对象
    """

    def __init__(self, node) -> None:
        self.name = node.name if node.name else ""
        self.docstring = ast.get_docstring(node)
        self.functions: list[FunctionNode] = []
        self.inner_classes: list[ClassNode] = []


class FileNode:
    """
    py文件信息保存对象
    """

    def __init__(self, name):
        self.name = name
        self.docstring = ""
        self.functions: list[FunctionNode] = []
        self.classes: list[ClassNode] = []


class ExtractNode(ABC):
    """
    从解析的对象中提取函数信息，通过重写_extract_func_node来执行想要的操作
    """

    @abstractmethod
    def _extract_func_node(self, class_name: str, functions: list[FunctionNode], obj: any):
        """
        需要自己实现相关逻辑
        :param class_name: 类名，即functions中保存函数信息所属的类名
        :param functions: 函数信息，保存某个类中所有的函数信息
        :param obj: 用于保存数据的对象
        :return:
        """
        pass

    def _extract_class_node(self, class_name: str, classes: list[ClassNode], obj: any):
        for cla in classes:
            self._extract_class_node(f"{class_name}.{cla.name}" if class_name != "" else cla.name, cla.inner_classes,
                                     obj)
            self._extract_func_node(f"{class_name}.{cla.name}" if class_name != "" else cla.name, cla.functions, obj)

    def _extract_file_node(self, file_node: FileNode, obj: any):
        if file_node.name != "":
            self._extract_func_node("", file_node.functions, obj)
            self._extract_class_node("", file_node.classes, obj)

    def extract(self, file_node: FileNode, obj: any):
        self._extract_file_node(file_node, obj)


class Visitor(ast.NodeVisitor):
    """
    继承ast.NodeVisitor，自动解析不同类型的节点，从而实现不同的操作，
    这里是对自己想实现功能的包装，不对外开放。
    """

    def __init__(self, name):
        self.file = FileNode(name)

    def visit_Expr(self, node):
        # 确保节点有值，并且是字符串类型
        if isinstance(node.value, ast.Str):
            # 提取文档字符串
            self.file.docstring = node.value.s

    def visit_ClassDef(self, node):
        # 解析内部类
        self.file.classes.append(generate_class(node))

    def visit_FunctionDef(self, node):
        self.file.functions.append(generate_func(node))


def parse_code(code_str, file_name) -> FileNode:
    """
    解析py文件，从中提取函数信息
    :param code_str:
    :param file_name:
    :return:
    """
    try:
        tree = ast.parse(code_str)
        file_visitor = Visitor(file_name)
        file_visitor.visit(tree)
        return file_visitor.file
    except Exception as e:
        return FileNode("")


def parse_func_code(code_str) -> FunctionNode:
    """
    只解析函数
    :param code_str:
    :return:
    """
    try:
        tree = ast.parse(code_str)
        return generate_func(tree.body[0])
    except Exception as e:
        return None
