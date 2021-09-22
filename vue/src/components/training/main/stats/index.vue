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
      <template v-for="(output, key) of statisticData">
        <template v-for="(item, i) of output">
          <Heatmap v-if="selected.includes(key) && item.type === 'Heatmap'" v-bind="item" :key="`heatmap_${i}`" />
          <Table v-if="selected.includes(key) && item.type === 'Table'" v-bind="item" :key="`table_${i}`" />
          <Scatter v-if="selected.includes(key) && item.type === 'Scatter'" v-bind="item" :key="`scatter_${i}`" />
          <Graphic v-if="selected.includes(key) && item.type === 'Graphic'" v-bind="item" :key="`graphic_${i}`" />
          <Histogram
            v-if="selected.includes(+key) && item.type === 'Histogram'"
            v-bind="item"
            :key="'Histogram' + i"
          />
        </template>
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
      return this.$store.getters['trainings/getTrainData']('statistic_data') || [];
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
      console.log(key)
      console.log(typeof key)

      this.selected = !this.selected.includes(key)
        ? [...this.selected, key]
        : this.selected.filter(item => item !== key);
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
