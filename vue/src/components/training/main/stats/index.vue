<template>
  <div class="t-scatters">
    <div class="t-scatters__header">
      <div class="t-scatters__checks">
        <template v-for="(item, i) of outputLayers">
          <t-field :key="'check_' + i" inline :label="`Выходной слой «${item}»`">
            <t-checkbox-new small :name="`${item}`" @change="change(item)" />
          </t-field>
        </template>
      </div>
      <t-field inline :label="`Автообновление`">
        <t-checkbox-new v-model="auto" small @change="autoChange" />
      </t-field>
      <div class="t-scatters__btn">
        <t-button @click="handleClick">Показать</t-button>
      </div>
    </div>
    <div class="t-scatters__content">
      <template v-for="(item, i) of filtesLayers">
        <component :is="item.type" v-bind="item" :key="`${item.type + i}`" />
      </template>
    </div>
  </div>
</template>

<script>
export default {
  name: 't-scatters',
  components: {
    Heatmap: () => import('./Heatmap'),
    Scatter: () => import('./Scatter'),
    Histogram: () => import('./Histogram'),
    Table: () => import('./Table'),
    Graphic: () => import('./Graphic'),
  },
  props: {
    outputs: Array,
  },
  computed: {
    statisticData() {
      return this.$store.getters['trainings/getTrainData']('statistic_data') || {};
    },
    filtesLayers() {
      return Object.entries(this.statisticData)
        .filter(item => this.selected.includes(+item[0]))
        .map(item => item[1]);
    },
    outputLayers() {
      return this.outputs.map(item => item.id);
    },
  },
  data: () => ({
    selected: [],
    auto: false,

  }),
  methods: {
    // isShow(layer, type) {
    //   ершыюisShowKeys.includes(+layer) && type === 'Heatmap';
    // },
    change(key) {
      // console.log(key)

      this.selected = !this.selected.includes(key)
        ? [...this.selected, key]
        : this.selected.filter(item => item !== key);
      console.log(this.selected);
    },
    async handleClick() {
      const data = {
        statistic_data: {
          output_id: this.selected,
          autoupdate: this.auto,
        },
      };

      await this.$store.dispatch('trainings/interactive', data);
    },
    autoChange(e) {
      this.auto = e.value;
    },
  },
};
</script>

<style lang="scss" scoped>
.t-scatters {
  position: relative;
  margin-bottom: 20px;
  &__header {
    display: flex;
    gap: 25px;
  }
  &__checks {
    display: flex;
    flex-wrap: wrap;
    flex-shrink: 0;
    max-width: 320px;
  }
  &__btn {
    margin-left: auto;
    flex: 0 0 150px;
  }
  &__content {
    display: flex;
    flex-wrap: wrap;
    gap: 50px;
  }
}
</style>
