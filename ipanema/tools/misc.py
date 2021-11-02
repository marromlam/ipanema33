################################################################################
#                                                                              #
#                                  MISCELANEA                                  #
#                                                                              #
################################################################################

import ast, math

__all__ = ["dig_tree"]

################################################################################
# Get vars from string expression ##############################################

class IdentifierExtractor(ast.NodeVisitor):
  def __init__(self):
    self.ids = set()
  def visit_Name(self, node):
    self.ids.add(node.id)

def get_vars_from_string(FUN):
  extractor = IdentifierExtractor()
  extractor.visit(ast.parse(FUN))
  extractor.ids = extractor.ids - set({**vars(math),**{'alpha':0}})
  return list(extractor.ids)

################################################################################



################################################################################
# Root file digger #############################################################

def dig_tree(file, ident = ""):
  for k,v in file.items():
    ident +='  '
    if str(v)[1:8] == 'TBranch':
      print(ident+"* {0}".format( k.decode() ))
    else:
      print(ident+'* '+k.decode())
      dig_tree(v,ident+'  ')
    ident = ident[:-2]

################################################################################
