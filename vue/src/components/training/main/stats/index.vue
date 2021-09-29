<template>
  <div class="t-scatters">
    <LoadSpiner class="overlay" v-show="isPending" text="Обновление..." />
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
      <template v-for="(layer, index) of statisticData">
        <template v-for="(data, i) of layer">
          <component v-if="selected.includes(+index)" :is="data.type" v-bind="data" :key="`${data.type + i + index}`" />
        </template>
      </template>
    </div>
  </div>
</template>

<script>
import LoadSpiner from '@/components/forms/LoadSpiner';

export default {
  name: 't-scatters',
  components: {
    Heatmap: () => import('./Heatmap'),
    Scatter: () => import('./Scatter'),
    Histogram: () => import('./Histogram'),
    Table: () => import('./Table'),
    Graphic: () => import('./Graphic'),
    LoadSpiner
  },
  props: {
    outputs: Array,
  },
  computed: {
    statisticData() {
      return this.$store.getters['trainings/getTrainData']('statistic_data') || {};
    },
    outputLayers() {
      return this.outputs.map(item => item.id);
    },
  },
  data: () => ({
    selected: [],
    auto: false,
    isPending: false
  }),
  methods: {
    change(key) {
      this.selected = !this.selected.includes(key)
        ? [...this.selected, key]
        : this.selected.filter(item => item !== key);
      console.log(this.selected);
    },
    async handleClick() {
      this.isPending = true
      const data = {
        statistic_data: {
          output_id: this.selected,
          autoupdate: this.auto,
        },
      };
      await this.$store.dispatch('trainings/interactive', data);
      this.isPending = false
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
  min-height: 200px;
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
  .overlay {
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background-color: rgb(14 22 33 / 30%);
    z-index: 5;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-direction: column;
  }
}
</style>
