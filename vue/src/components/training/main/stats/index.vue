<template>
  <div class="t-scatters">
    <div class="t-scatters__header">
      <div class="checks">
        <template v-for="(item, i) of outputLayers">
          <t-checkbox
            :key="'check_' + i"
            :inline="true"
            :label="`Выходной слой «${item.id}»`"
            :name="`${item.id}`"
            @change="change($event, item.id)"
          />
        </template>
        <t-checkbox :inline="true" v-model="auto" @change="autoChange" label="Автообновление" />
      </div>
      <t-button class="t-scatters__btn" @click="handleClick">Показать</t-button>
    </div>
    <div class="t-scatters__content">
      <template v-for="(output, key, i) of statisticData">
        <component :is="'Heatmap'"
        v-if="isShowKeys.includes(+key)"
        v-bind="output"
        :key="i" />
      </template>
      <Scatter />
      <Histogram />
      <Table />
      <Graphic />
    </div>
  </div>
</template>

<script>
import Heatmap from './Heatmap.vue';
import Scatter from './Scatter.vue';
import Histogram from './Histogram.vue';
import Table from './Table.vue';
import Graphic from './Graphic.vue';

export default {
  name: 't-scatters',
  components: {
    Heatmap,
    Scatter,
    Histogram,
    Table,
    Graphic
  },
  computed: {
    statisticData() {
      return this.$store.getters['trainings/getTrainData']('statistic_data') || [];
    },
    outputLayers() {
      const layers = this.$store.getters['modeling/getModel'].layers
      if (!layers) return []
      return layers.filter(item => item.group === 'output')
    }
  },
  data: () => ({
    isShowKeys: [],
    auto: false
  }),
  methods: {
    change(e, key) {
      this.isShowKeys = !this.isShowKeys.includes(key)
        ? [...this.isShowKeys, key]
        : this.isShowKeys.filter(item => item !== key);
    },
    async handleClick() {
      const data = {
        "statistic_data": {
          "output_id": this.isShowKeys,
          "autoupdate": this.auto
        }
      }
      this.$store.dispatch('trainings/setTrainDisplay', data)

      await this.$store.dispatch('trainings/interactive', this.$store.getters['trainings/getTrainDisplay'])
    },
    autoChange(e) {
      this.auto = e.value
    }
  }
};
</script>

<style lang="scss" scoped>
.t-scatters {
  &__header {
    display: flex;
    gap: 25px;
    .checks {
      display: flex;
      flex-wrap: wrap;
      flex-shrink: 0;
      max-width: 320px;
    }
    button {
      flex: 0 0 150px;
    }
  }
  &__content {
    display: flex;
    flex-wrap: wrap;
    gap: 50px;
  }
}
</style>