<template>
  <div class="t-graphics">
    <div class="t-graphics__button">
      <scrollbar>
        <div class="t-graphics__extra">
          <div
            v-for="(value, i) of predict"
            :key="'t-graphics' + i"
            class="t-graphics__item"
            :class="i == name ? 't-graphics__item--active' : ''"
            @click="name = i"
          >
            {{ key }}
          </div>
        </div>
      </scrollbar>
    </div>
    <div class="t-graphics__main">
      <Plotly :data="testPredictData(predict[name])" :layout="layout()" :display-mode-bar="false" />
      <div class="t-graphics__title">{{ name }}</div>
    </div>
  </div>
</template>

<script>
import { Plotly } from 'vue-plotly';

export default {
  name: 'GraphicCard',
  components: {
    Plotly,
  },
  props: {
    source: Object,
    predict: Object
  },
  data: () => ({
    name: null,
    line: {
      dash: "dot",
      width: 4,
      color: '#89D764'
    },
    test: {
      plot_bgcolor: "#d3d3d3",
      paper_bgcolor: "#d3d3d3"
    },
    defLayout: {
      height: 350,
      width: 600,
      plot_bgcolor: '#fff0',
      paper_bgcolor: '#242F3D',
      showlegend: true,
      legend: {
        orientation: 'h',
      },
      font: {
        color: '#A7BED3',
        size: 9,
      },
      margin: {
        pad: 1,
        t: 5,
        r: 5,
        b: 30,
        l: 30,
      },
      xaxis: {
        gridcolor: '#17212B',
        showline: false,
        linecolor: '#A7BED3',
        linewidth: 1,
        title: {
          // text: 'Время',
          standoff: 0,
          font: { size: 10 },
        },
      },
      yaxis: {
        gridcolor: '#17212B',
        showline: false,
        linecolor: '#A7BED3',
        linewidth: 1,
        title: {
          // text: 'Значение',
          font: { size: 10 },
        },
      },
    },
  }),
  methods: {
    layout() {
      const layout = this.defLayout;
      layout.xaxis.title.text = '';
      layout.yaxis.title.text = '';
      return layout;
    },
    testPredictData(tag){
      let index = 0;
      let plot_data = [];
      for(let index_arr in tag){
        let arr = tag[index_arr];
        let plot_line = {x: [], y: [], type: 'scatter', name: 'Предсказанные значения'};
        if(index > 0)plot_line['line'] = this.line;
        else plot_line['name'] = 'Реальные значения'
        for(let i in arr){
          plot_line['x'].push(index);
          plot_line['y'].push(arr[i]);
          index++;
        }
        index--;
        plot_data.push(plot_line);
      }
      return plot_data;
    },
  },
  mounted() {
    this.name = Object.keys(this.predict)[0] || null
  }
};
</script>

<style lang="scss" scoped>
.t-graphics {
  height: 400px;
  width: 630px;
  display: flex;
  flex-direction: column;
  user-select: none;
  &__title {
    position: absolute;
    top: 0;
    right: calc(50% - 30px);
    color: #7e7e7e;
  }
  &__main {
    flex: 1 1 auto;
    border: 1px solid #6c7883;
    border-radius: 4px;
    position: relative;
  }
  &__extra {
    //flex: 0 0 110px;
    //padding: 0 7px;
    //height: 230px;
    display: flex;
  }
  &__item {
    height: 40px;
    width: 100px;
    color: #a7bed3;
    background-color: #242f3d;
    border: 1px solid #6c7883;
    border-radius: 4px;
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: center;
    transition: all 0.2s ease-in-out;
    &:hover {
      background-color: #65b9f4;
    }
    &--active{
      border-color: #65B9F4;
      color: #ebeeee;
    }
  }
}
</style>
