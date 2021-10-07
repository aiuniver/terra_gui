<template>
  <div class="t-scatters">
    <div class="t-scatters__header">
      <div class="t-scatters__checks">
        <template v-for="({ id, value }, i) of outputLayers">
          <t-field :key="'check_' + i" inline :label="`Выходной слой «${id}»`">
            <t-checkbox-new small :value="value" @change="change({ id, value: $event.value })" />
          </t-field>
        </template>
      </div>
      <t-field inline :label="`Автообновление`">
        <t-checkbox-new v-model="settings.autoupdate" small @change="send" />
      </t-field>
    </div>
    <div class="t-scatters__content">
      <template v-for="(layer, index) of statisticData">
        <template v-for="(data, i) of layer">
          <component v-if="ids.includes(+index)" :is="type[data.type]" v-bind="data" :key="`${data.type + i + index}`" />
        </template>
      </template>
      <LoadSpiner
        v-if="isLearning && ids.length && !Object.keys(statisticData).length"
        class="overlay"
        text="Загрузка данных..."
      />
    </div>
  </div>
</template>

<script>
import LoadSpiner from '@/components/forms/LoadSpiner';
import { mapGetters } from 'vuex';
export default {
  name: 't-scatters',
  components: {
    Heatmap: () => import('./Heatmap'),
    Scatter: () => import('./Scatter'),
    Histogram: () => import('./Histogram'),
    Table: () => import('./Table'),
    Graphic: () => import('./Graphic'),
    LoadSpiner,
  },
  props: {
    outputs: Array,
  },
  data: () => ({
    type: {
      heatmap: 'heatmap',
      'correlation heatmap': 'CorrelationHeatmap',
      scatter: 'scatter',
      'distribution histogram': 'histogram',
      histogram: 'histogram',
      table: 'table',
      graphic: 'graphic',
    },
  }),
  computed: {
    ...mapGetters({
      status: 'trainings/getStatus',
    }),
    isLearning() {
      return ['addtrain', 'training'].includes(this.status);
    },
    settings: {
      set(value) {
        this.$store.dispatch('trainings/setObjectInteractive', { statistic_data: value });
      },
      get() {
        return this.$store.getters['trainings/getObjectInteractive']('statistic_data');
      },
    },
    statisticData() {
      return this.$store.getters['trainings/getTrainData']('statistic_data') || {};
    },
    outputLayers() {
      return this.outputs.map(item => {
        return {
          id: item.id,
          value: this.ids.includes(item.id),
        };
      });
    },
    ids() {
      return JSON.parse(JSON.stringify(this.settings.output_id || []));
    },
  },
  methods: {
    change({ id, value }) {
      this.settings.output_id = value ? [...this.ids, id] : [...this.ids.filter(item => item !== id)];
      this.send();
    },
    async send() {
      await this.$store.dispatch('trainings/interactive', {});
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
    position: relative;
    align-items: flex-start;
  }
  .overlay {
    width: 100%;
    height: 100%;
    z-index: 5;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-direction: column;
  }
}
</style>
