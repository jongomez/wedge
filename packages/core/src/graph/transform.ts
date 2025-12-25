import { ISignatureDef, ITFGraphDef } from "../external/tensorflow/compiled_api";
import { getNodeNameAndIndex } from "../external/tensorflow/executor_utils";
import { OperationMapper } from "../external/tensorflow/ops/operation_mappers";
import { WedgeBase } from "../types";
import { Graph } from "./types";

export function tfGraphToWedgeGraph(
  graph: ITFGraphDef,
  wedgeInstance: WedgeBase,
  signature: ISignatureDef = {},
): Graph {
  const tfNodes = graph.node;
  const placeholders: Node[] = [];
  const weights: Node[] = [];
  const initNodes: Node[] = [];

  const operationMapper = OperationMapper.Instance;

  const nodes = tfNodes.reduce<{ [key: string]: Node }>((map, node) => {
    map[node.name] = operationMapper.mapNode(node);
    if (node.op.startsWith('Placeholder')) {
      placeholders.push(map[node.name]);
    } else if (node.op === 'Const') {
      weights.push(map[node.name]);
    } else if (node.input == null || node.input.length === 0) {
      initNodes.push(map[node.name]);
    }
    return map;
  }, {});

  let inputs: Node[] = [];
  const outputs: Node[] = [];
  let inputNodeNameToKey: { [key: string]: string } = {};
  let outputNodeNameToKey: { [key: string]: string } = {};
  if (signature != null) {
    inputNodeNameToKey = operationMapper.mapSignatureEntries(signature.inputs);
    outputNodeNameToKey = operationMapper.mapSignatureEntries(signature.outputs);
  }
  const allNodes = Object.keys(nodes);
  allNodes.forEach(key => {
    const node = nodes[key];
    node.inputNames.forEach((name, index) => {
      const [nodeName, , outputName] = getNodeNameAndIndex(name);
      const inputNode = nodes[nodeName];
      if (inputNode.outputs != null) {
        const outputIndex = inputNode.outputs.indexOf(outputName);
        if (outputIndex !== -1) {
          const inputName = `${nodeName}:${outputIndex}`;
          // update the input name to use the mapped output index directly.
          node.inputNames[index] = inputName;
        }
      }
      node.inputs.push(inputNode);
      inputNode.children.push(node);
    });
  });

  // if signature has not outputs set, add any node that does not have
  // outputs.
  if (Object.keys(outputNodeNameToKey).length === 0) {
    allNodes.forEach(key => {
      const node = nodes[key];
      if (node.children.length === 0) {
        outputs.push(node);
      }
    });
  } else {
    Object.keys(outputNodeNameToKey).forEach(name => {
      const [nodeName,] = getNodeNameAndIndex(name);
      const node = nodes[nodeName];
      if (node != null) {
        node.signatureKey = outputNodeNameToKey[name];
        outputs.push(node);
      }
    });
  }

  if (Object.keys(inputNodeNameToKey).length > 0) {
    Object.keys(inputNodeNameToKey).forEach(name => {
      const [nodeName,] = getNodeNameAndIndex(name);
      const node = nodes[nodeName];
      if (node) {
        node.signatureKey = inputNodeNameToKey[name];
        inputs.push(node);
      }
    });
  } else {
    inputs = placeholders;
  }

  let functions = {};
  if (graph.library != null && graph.library.function != null) {
    functions = graph.library.function.reduce((functions, func) => {
      functions[func.signature.name] = operationMapper.mapFunction(func);
      return functions;
    }, {} as { [key: string]: Graph });
  }

  const result: Graph =
    { nodes, inputs, outputs, weights, placeholders, signature, functions };

  if (initNodes.length > 0) {
    result.initNodes = initNodes;
  }

  return result;
}

