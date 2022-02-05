<template>
  <div class="t-statistic">
    <div class="t-statistic__header">
      <div class="t-statistic__checks">
        <template v-for="({ id, value }, i) of outputLayers">
          <t-field :key="'check_' + i" inline :label="`Выходной слой «${id}»`">
            <TCheckbox small :value="value" @change="change({ id, value: $event.value })" />
          </t-field>
        </template>
      </div>
      <div v-if="isYolo">
        <t-field inline label="Чувствительность">
          <TInputNew v-model.number="settings.sensitivity" type="number" small style="width: 109px" @change="send" />
        </t-field>
        <t-field inline label="Порог отображения">
          <TInputNew v-model.number="settings.threashold" type="number" small style="width: 109px" @change="send" />
        </t-field>
        <t-field inline label="Бокс-канал">
          <TSelect :list="numOutput" v-model="settings.box_channel" small @change="send" />
        </t-field>
      </div>
      <div>
        <t-field inline :label="`Автообновление`">
          <TCheckbox v-model="settings.autoupdate" small @change="send" />
        </t-field>
      </div>
    </div>
    <div class="t-statistic__content">
      <template v-for="(layer, index) of statisticData">
        <component :is="component(layer.type)" v-bind="layer" :key="`${'layer.type' + index}`" />
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
import TSelect from '@/components/new/forms/TSelect';
import TCheckbox from '@/components/new/forms/TCheckbox';
import TInputNew from '@/components/new/forms/TInputNew';
import { mapGetters } from 'vuex';

export default {
  name: 't-statistic',
  components: {
    Colormap: () => import('../stats/Colormap'),
    Heatmap: () => import('../stats/Heatmap'),
    Corheatmap: () => import('../stats/Corheatmap'),
    Valheatmap: () => import('../stats/Valheatmap'),
    Scatter: () => import('../stats/Scatter'),
    Histogram: () => import('../stats/Histogram'),
    Bar: () => import('../stats/Histogram'),
    STable: () => import('../stats/STable'),
    Graphic: () => import('../stats/Graphic'),
    LoadSpiner,
    TSelect,
    TCheckbox,
    TInputNew
  },
  props: {
    outputs: Array,
  },
  computed: {
    ...mapGetters({
      status: 'trainings/getStatus',
      architecture: 'trainings/getArchitecture',
    }),
    isYolo() {
      return ['YoloV4', 'YoloV3'].includes(this.architecture);
    },
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
    numOutput() {
      return this.outputs.map((_, i) => {
        return {
          label: `${i}`,
          value: i,
        };
      });
    },
    ids() {
      return JSON.parse(JSON.stringify(this.settings.output_id || []));
    },
  },
  methods: {
    component(comp) {
      return comp === 'table' ? 's-table' : comp;
    },
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
.t-statistic {
  position: relative;
  margin-bottom: 20px;
  &__header {
    display: flex;
    gap: 25px;
  }
  &__checks {
    // display: flex;
    // flex-wrap: wrap;
    // flex-shrink: 0;
    // max-width: 320px;
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
