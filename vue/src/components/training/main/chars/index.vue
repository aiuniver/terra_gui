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
    outputs: Array,
    interactive: Object,
  },
  data: () => ({
    // charts: [],
    menus: [
      {
        name: 'Показывать данные',
        list: [
          { title: 'По всей модели', event: { name: 'data', data: 'model' } },
          { title: 'По классам', event: { name: 'data', data: 'classes' } },
        ],
      },
    ],
  }),
  computed: {
    ...mapGetters({
      chars: 'trainings/getChars',
    }),
    charts: {
      set(value) {
        this.$store.dispatch('trainings/setCharts', value)
      },
      get() {
        return this.$store.getters['trainings/getCharts'](this.metric)
      }
    },
    data() {
      return this.$store.getters['trainings/getTrainData'](this.metric) || [];
    },
    allMenus() {
      // console.log([...this.menus, this.listChats]);
      // console.log(this.listChats);
      return [...this.menus, ...this.listChats];
    },
    listChats() {
      const arr = [];
      const listOutputs = this.outputs.map(item => {
        return { title: `Выход ${item.id}`, event: { name: 'chart', data: item.id } };
      });

      if (this.metric === 'metric_graphs') {
        const listMetrics = this.metrics.map(item => {
          return { title: `${item}`, event: { name: 'metric', data: item } };
        });
        arr.push({
          name: 'Показывать метрики',
          list: listMetrics,
        });
      }
      arr.push({
        name: 'Показывать выход',
        list: listOutputs,
      });
      return arr;
    },
    outputIdx() {
      return this.outputs.find(item => item.id).id;
    },
    metrics() {
      return this.outputs?.[0].metrics ?? [];
    },
  },
  mounted() {
    // console.log(this.outputs);
    // const data = this.interactive?.[this.metric] || [];
    // this.charts = data.map(item => item);
    // this.charts = this.interactive?.[this.metric]
  },
  methods: {
    event({ name, data }, { id }) {
      console.log(name, data, id);
      if (data === 'add') {
        this.add();
      }
      if (data === 'remove') {
        this.remove(id);
      }
      if (data === 'copy') {
        this.copy(id);
      }
      if (name === 'chart') {
        this.charts = this.charts.map(item => {
          if (item.id === id) {
            item.output_idx = data;
          }
          return item;
        });
        this.send(this.charts);
      }
      if (name === 'data') {
        this.charts = this.charts.map(item => {
          if (item.id === id) {
            item.show = data;
          }
          return item;
        });
        this.send(this.charts);
      }
      if (name === 'metric') {
        this.charts = this.charts.map(item => {
          if (item.id === id) {
            item.show_metric = data;
          }
          return item;
        });
        this.send(this.charts);
      }
    },
    getChart({ id }) {
      // console.log({ graphID });
      return this.data.find(item => item.id === id);
    },
    add() {
      if (this.charts.length < 10) {
        let maxID = Math.max(0, ...this.charts.map(o => o.id));
        if (this.metric === 'metric_graphs') {
          this.charts.push({ id: maxID + 1, output_idx: 2, show: 'model', show_metric: this.metrics[0] });
        } else {
          this.charts.push({ id: maxID + 1, output_idx: 2, show: 'model' });
        }
        this.send(this.charts);
      }
    },
    copy(id) {
      if (this.charts.length < 10) {
        let maxID = Math.max(0, ...this.charts.map(o => o.id));
        const char = { ...this.charts.find(item => item.id === id) };
        if (char) {
          char.id = maxID + 1;
          this.charts = [...this.charts, char];
        }
        this.send(this.charts);
      }
    },
    remove(id) {
      this.charts = this.charts.filter(item => item.id !== id);
      this.send(this.charts);
    },
    async send(data) {
      const res = await this.$store.dispatch('trainings/interactive', { [this.metric]: data });
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