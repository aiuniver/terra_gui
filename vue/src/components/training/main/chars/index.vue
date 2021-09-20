<template>
  <div class="t-charts">
    <!-- <div class="charts__title">Графики</div> -->
    <div class="t-charts__content">
      <TCharTemp class="t-chart" @click.native="add" />
      <TChar
        v-for="(settings, i) of charts"
        v-bind="getChart(settings)"
        class="t-chart"
        :key="'char1_' + i"
        :settings="settings"
        :menus="allMenus"
        @event="event($event, settings)"
      />
    </div>
    <div v-if="!data.length" class="t-charts__empty">Нет данных</div>
  </div>
</template>

<script>
import TChar from './TChar';
import TCharTemp from './TCharTemp';
import { mapGetters } from 'vuex';
export default {
  name: 't-graphs',
  components: {
    TChar,
    TCharTemp,
  },
  props: {
    metric: String,
  },
  data: () => ({
    charts: [],
    menus: [
      {
        name: 'Показывать данные',
        list: [
          { title: 'По всей модели', event: { name: 'data', data: 'model' } },
          { title: 'По классам', event: { name: 'data', data: 'classes' } },
        ],
      },
      // {
      //   name: 'Показывать метрики',
      //   list: [
      //     { title: 'Accuracy', event: { name: 'metric', data: 'Accuracy' } },
      //     { title: 'Hinge', event: { name: 'metric', data: 'Hinge' } },
      //   ],
      // },
    ],
  }),
  computed: {
    ...mapGetters({
      chars: 'trainings/getChars',
    }),
    // filstrCharts() {
    //   return this.data.filter((chart) => this.charts.includes(chart.id));
    // },
    data() {
      return this.$store.getters['trainings/getTrainData'](this.metric) || [];
    },
    allMenus() {
      console.log([...this.menus, this.listChats]);
      return [...this.menus, this.listChats];
    },
    listChats() {
      const list = this.data.map(item => {
        return { title: item.graph_name, event: { name: 'chart', data: item.id } };
      });
      return {
        name: 'Выходы',
        list,
      };
    },
  },
  mounted() {},
  methods: {
    event({ name, data }, { id, graphID }) {
      // console.log(name, data, id);
      if (data === 'remove') {
        this.remove(id);
      }
      if (name === 'chart') {
        this.charts = this.charts.map(item => {
          if (item.id === id) {
            item.graphID = data;
          }
          return item;
        });
      }
      if (name === 'data') {
        this.send({
          lost_graphs: {
            id: id,
            output_idx: graphID,
            show: data,
          },
        });
      }
    },
    getChart({ id }) {
      // console.log({ graphID });
      return this.data.find(chart => chart.id === id);
    },
    add() {
      if (this.charts.length < 10) {
        // const chart = this.data.find(item => item);
        // if (chart) {
        let maxID = Math.max(0, ...this.charts.map(o => o.id));
        this.charts.push({ id: maxID + 1, output_idx: 2, show: 'model' });
        this.send(this.charts);
        // }
      }
    },
    remove(id) {
      this.charts = this.charts.filter(item => item.id !== id);
      // console.log(id);
    },
    async send(data) {
      const res = await this.$store.dispatch('trainings/interactive', { lost_graphs: data });
      console.log(`response`, res);
    },
  },
};
</script>

<style lang="scss" scoped>
.t-chart {
  width: 48%;
  margin: 0 0 20px 0;
}
.t-charts {
  margin-bottom: 20px;

  // &__title {
  // }
  &__content {
    margin-top: 10px;
    display: flex;
    flex-direction: row;
    flex-wrap: wrap;
    justify-content: flex-start;
    align-content: flex-start;
    align-items: flex-start;
    gap: 2%;
  }
  &__empty {
    display: flex;
    height: 100%;
    justify-content: center;
    padding-top: 10px;
    font-size: 16px;
    opacity: 0.5;
    user-select: none;
  }
}
</style>