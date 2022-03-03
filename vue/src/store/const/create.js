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

const chengeParametrs = (data) => {
  const options = { ...data }
  delete options.type
  return {
    type: data.type,
    options
  }
}

const isError = (res) => {
  if (!res.success) return true;
  const errors = res?.data || {}
  for (const key in errors) {
    if (errors[key]) return true;
  }
  return false;
}

const chnageType = (data) => {
  const arr = JSON.parse(JSON.stringify(data))
  return arr.map(i => {
    // if (['input', 'output'].includes(i.type)) i.type = 'layer'
    if (i.type === 'handler') i.parameters = { ...chengeParametrs(i.parameters) }
    return i
  })
}

const createObj = (data) => {
  const { project, inputs, outputs } = JSON.parse(JSON.stringify(data))
  const { name, architecture, tags, verName, train, shuffle, source, firstCreation } = project
  return {
    name,
    source,
    architecture,
    tags,
    first_creation: firstCreation,
    version: {
      name: verName,
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


export { getFiles, createObj, chnageType, isError }