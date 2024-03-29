<template>
  <div class="predictions">
    <!-- <h3>Параметры</h3> -->
    <div class="predictions__params">
      <div class="predictions__param">
        <t-field inline label="Данные для расчета">
          <DSelect :list="sortData" v-model="example_choice_type" small @change="show" />
        </t-field>
        <t-field inline label="Тип выбора данных">
          <DSelect :list="sortOutput" v-model="main_output" small @change="show" />
        </t-field>
        <t-field inline label="Тип выборки">
          <DSelect v-model="data_type" :list="list" small @change="show" />
        </t-field>
        <t-field inline label="Показать примеров">
          <DInputNumber
            v-model.number="num_examples"
            type="number"
            small
            style="width: 109px"
            :error="isError"
            @change="show"
          />
        </t-field>
      </div>
      <div v-if="isYolo" class="predictions__param">
        <t-field inline label="Бокс-канал">
          <DSelect :list="numOutput" v-model="box_channel" small @change="show" />
        </t-field>
        <t-field inline label="Чувствительность">
          <DInputNumber v-model="sensitivity" type="number" small style="width: 109px" @change="show" />
        </t-field>
        <t-field inline label="Порог отображения">
          <DInputNumber v-model="threashold" type="number" small style="width: 109px" @change="show" />
        </t-field>
      </div>
      <div class="predictions__param">
        <t-field inline label="Выводить промежуточные результаты" :key="'show_results' + show_results">
          <DCheckbox v-model="show_results" small @change="show" />
        </t-field>
        <t-field inline label="Показать статистику" :key="'show_statistic' + show_statistic">
          <DCheckbox v-model="show_statistic" small @change="show" />
        </t-field>
        <!-- <t-field inline label="Фиксация колонок">
          <Checkbox v-model="fixation" small />
        </t-field> -->
      </div>
      <div class="predictions__param"></div>
      <div class="predictions__param">
        <DButton style="width: 150px" @click.native="show" :disabled="!!isError">
          {{ 'Обновить' }}
        </DButton>
        <br />
        <t-field inline label="Автообновление" :key="'autoupdate' + autoupdate">
          <DCheckbox v-model="autoupdate" small @change="show" />
        </t-field>
      </div>
    </div>
    <div class="predictions__body">
      <PredictTable v-if="isEmpty" :predict="predictData" :fixation="fixation" :update="predictUpdate" />
      <div v-else class="predictions__overlay">
        <LoadSpiner v-if="isStart && isLearning" text="Загрузка данных..." />
      </div>
    </div>
  </div>
</template>

<script>
import { mapGetters } from 'vuex';

export default {
  name: 'Predictions',
  components: {
    PredictTable: () => import('./PredictTable'),
    LoadSpiner: () => import('@/components/forms/LoadSpiner'),
  },
  props: {
    outputs: Array,
    interactive: Object,
  },
  data: () => ({
    start: false,
    sortData: [
      { label: 'Best', value: 'best' },
      { label: 'Worst', value: 'worst' },
      { label: 'Seed', value: 'seed' },
      { label: 'Random', value: 'random' },
    ],
    list: [
      { label: 'Тренировочная', value: 'train' },
      { label: 'Проверочная', value: 'val' },
    ],
    autoupdate: false,
    example_choice_type: 'seed',
    main_output: 2,
    data_type: 'val',
    num_examples: 10,
    show_results: false,
    show_statistic: false,
    fixation: false,
    max: 10,
    sensitivity: 0.12,
    box_channel: 1,
    threashold: 0.1,
  }),
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
    isStart() {
      // return this.settings.show_results;
      return this.settings.show_results && this.autoupdate;
    },
    isEmpty() {
      return Boolean(Object.keys(this.predictData).length);
    },
    sortOutput() {
      return this.outputs.map(item => {
        return {
          label: `Выходной слой ${item.id}`,
          value: item.id,
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
    update() {
      for (let key in this.settings) {
        if (this[key] !== this.settings[key]) {
          return true;
        }
      }
      return false;
    },
    isError() {
      const value = this.num_examples;
      if (value === '') {
        return 'Поле не может быть пустым';
      }
      if (value > 10) {
        return 'Не больше 10 примеров';
      }
      if (value < 1) {
        return 'Не меньше 1 примера';
      }
      return '';
    },
    settings() {
      return this.$store.getters['trainings/getObjectInteractive']('intermediate_result');
    },
    predictData() {
      return this.$store.getters['trainings/getTrainData']('intermediate_result') || {};
    },
    predictUpdate() {
      return this.$store.getters['trainings/getTrainData']('update') || '';
    },
    statusTrain() {
      return this.$store.getters['trainings/getStatusTrain'];
    },
  },
  methods: {
    async show(val) {
      console.log('val', val)
      if (this.isError) return;
      // this.start = this.settings.show_results;
      const data = {
        autoupdate: this.autoupdate,
        example_choice_type: this.example_choice_type,
        main_output: this.main_output,
        num_examples: this.num_examples > 10 ? 10 : this.num_examples,
        show_results: this.show_results,
        show_statistic: this.show_statistic,
        data_type: this.data_type,
        sensitivity: this.sensitivity,
        box_channel: this.box_channel,
      };
      console.log('data interactive', data)
      await this.$store.dispatch('trainings/interactive', { intermediate_result: data });
    },
  },
  created() {
    for (let key in this.settings) {
      this[key] = this.settings[key];
    }
    this.start = this.settings.show_results && this.autoupdate;
  },
  watch: {
    settings(value) {
      for (let key in value) {
        this[key] = value[key];
      }
      console.warn(value['show_results']);
    },
  },
};
</script>

<style lang="scss" scoped>
.predictions {
  position: relative;
  &__body {
    position: relative;
    width: 100%;
  }
  &__overlay {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 100%;
    height: 100%;
    z-index: 5;
  }
  &__params {
    display: flex;
    margin-top: 10px;
    margin-bottom: 10px;
  }
  &__param {
    padding: 0 10px 0 0;
    height: 100%;
    &:last-child {
      margin-left: auto;
      padding: 0;
    }
  }
}
</style>
