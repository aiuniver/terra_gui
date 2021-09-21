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
    charts: [],
    menus: [
      {
        name: 'Показывать данные',
        list: [
          { title: 'По всей модели', event: { name: 'data', data: 'model' } },
          { title: 'По классам', event: { name: 'data', data: 'classes' } },
        ],
      },
      {
        name: 'Показывать метрики',
        list: [
          { title: 'Accuracy', event: { name: 'metric', data: 'Accuracy' } },
          { title: 'Hinge', event: { name: 'metric', data: 'Hinge' } },
        ],
      },
    ],
  }),
  computed: {
    ...mapGetters({
      chars: 'trainings/getChars',
    }),
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
        name: 'Показывать выход',
        list,
      };
    },
    outputIdx() {
      return this.outputs.find(item => item.id).id;
    },
  },
  mounted() {
    console.log(this.outputs);
    this.charts = this.interactive?.[this.metric] || [];
  },
  methods: {
    event({ name, data }, { id }) {
      // console.log(name, data, id);
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
            item.graphID = data;
          }
          return item;
        });
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
    },
    getChart({ id, chart }) {
      // console.log({ graphID });
      return this.data.find(item => item.id === chart) || this.data.find(item => item.id === id);
    },
    add() {
      if (this.charts.length < 10) {
        let maxID = Math.max(0, ...this.charts.map(o => o.id));
        this.charts.push({ id: maxID + 1, output_idx: 2, show: 'model', chart: 1, show_metric: 'Accuracy' });
        this.send(this.charts);
      }
    },
    copy(id) {
      if (this.charts.length < 10) {
        let maxID = Math.max(0, ...this.charts.map(o => o.id));
        const char = this.charts.find(item => item.id === id);
        if (char) {
          char.id = maxID + 2;
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