const test = [
  {
    type: "text",
    name: "text_name",
    parse: "inputs[name][parameters][text]",
    label: "Name text",
    default: ""
  },
  {
    type: "number",
    name: "word_to_vec_size",
    parse: "inputs[name][parameters][word_to_vec_size]",
    label: "Name number",
    default: 15,
    rule: "required",  // "trum", "email", "space"... ?? 
  },
  {
    type: "select",
    name: "select_size",
    parse: "inputs[name][parameters][select]",
    label: "Name select",
    default: "start",
    available: ["start", "center", "end"]
  }
]


// или 

const test = {
  "1": {
    type: "text",
    name: "text_name",
    parse: "inputs[name][parameters][text]",
    label: "Name text",
    default: ""
  },
  "2": {
    type: "number",
    name: "word_to_vec_size",
    parse: "inputs[name][parameters][word_to_vec_size]",
    label: "Name number",
    default: 15,
    rule: "required", 
  },
  // ....
}