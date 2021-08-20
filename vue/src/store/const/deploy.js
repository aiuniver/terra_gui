const defLayout = {
  autosize: true,
  margin: {
    l: 62,
    r: 20,
    t: 20,
    b: 67,
    pad: 0,
    autoexpand: true,
  },
  font: {
    color: "#A7BED3",
  },
  showlegend: true,
  legend: {
    y: -0.25,
    itemsizing: "constant",
    orientation: "h",
    font: {
      family: "Open Sans",
      color: "#A7BED3",
    },
  },
  paper_bgcolor: "transparent",
  plot_bgcolor: "transparent",
  title: {
    text: "",
  },
  xaxis: {
    title: "Эпоха",
    showgrid: true,
    zeroline: false,
    linecolor: "#A7BED3",
    gridcolor: "#0E1621",
    gridwidth: 1,
  },
  yaxis: {
    title: "accuracy",
    showgrid: true,
    zeroline: false,
    linecolor: "#A7BED3",
    gridcolor: "#0E1621",
    gridwidth: 1,
  },
}

const originaltextStyle = {
  width: "600px",
  height: "300px",
  color: "#A7BED3",
  padding: "10px 25px 12px 12px"
}

  export { defLayout, originaltextStyle };