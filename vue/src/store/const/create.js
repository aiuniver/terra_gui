const changeStructTable = (data) => {
  const newarr = [];
  data.forEach((el, index) => {
    el.forEach((elm, i) => {
      if (!newarr[i]) {
        newarr[i] = [];
      }
      newarr[i][index] = elm;
    });
  });
  return newarr;
}
const getFiles = (arr) => {
  return arr.map(e => {
    return {
      id: e.id,
      cover: e.cover,
      label: e.title,
      type: e.type,
      table: changeStructTable(e.data || []),
      value: e.path,
    };
  });
}

const chnageType = (arr) => {
  return arr.map(i => {
    if (['input', 'output'].includes(i.type)) i.type = 'layer'
    return i
  })
}

const createObj = (data) => {
  const { project, inputs, outputs, source_path } = JSON.parse(JSON.stringify(data))
  const { alias, name, task_type, tags, verAlias, verName, parent_alias, train, shuffle, datasets_path } = project
  return {
    alias,
    name,
    datasets_path,
    source_path,
    task_type,
    tags,
    version: {
      alias: verAlias,
      name: verName,
      datasets_path,
      parent_alias,
      info: {
        part: {
          train,
          validation: (100 - (train * 100)) / 100,
        },
        shuffle,
      },
      inputs: chnageType(inputs),
      outputs: chnageType(outputs)
    }
  }
}


export { getFiles, createObj }