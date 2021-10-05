<template>
  <div class="params">
    <div v-if="statusTrain === 'start'" class="params__overlay">
      <LoadSpiner :text="'Запуск обучения...'" />
    </div>
    <div class="params__body">
      <scrollbar>
        <div class="params__items">
          <at-collapse :value="collapse">
            <at-collapse-item class="mt-3" :title="''">
              <template v-for="(data, i) of main.fields">
                <t-auto-field-trainings
                  v-bind="data"
                  :key="'main_' + i"
                  :state="state"
                  :inline="false"
                  :disabled="disabledAny"
                  @parse="parse"
                />
              </template>
            </at-collapse-item>
            <at-collapse-item class="mt-3" :title="''">
              <div class="fit">
                <template v-for="(data, i) of fit.fields">
                  <t-auto-field-trainings
                    v-bind="data"
                    :key="'fit_' + i"
                    class="fit__item"
                    :state="state"
                    :inline="true"
                    :disabled="disabledAny"
                    @parse="parse"
                  />
                </template>
              </div>
            </at-collapse-item>
            <at-collapse-item class="mt-3" :title="optimizer.name">
              <div class="optimizer">
                <template v-for="(data, i) of optimizerFields">
                  <t-auto-field-trainings
                    v-bind="data"
                    :key="'optimizer_' + i + data.parse"
                    class="optimizer__item"
                    :state="state"
                    inline
                    :disabled="disabledAny"
                    @parse="parse"
                  />
                </template>
              </div>
            </at-collapse-item>
            <at-collapse-item class="mt-3" :title="outputs.name">
              <div class="blocks-layers">
                <template v-for="(field, i) of outputs.fields">
                  <div class="block-layers" :key="'block_layers_' + i">
                    <div class="block-layers__header">
                      {{ field.name }}
                    </div>
                    <div class="block-layers__body">
                      <template v-for="(data, i) of field.fields">
                        <t-auto-field-trainings
                          v-bind="data"
                          :key="'checkpoint_' + i + data.parse"
                          :state="state"
                          :inline="true"
                          :disabled="data.disabled || disabled"
                          @parse="parse"
                        />
                      </template>
                    </div>
                  </div>
                </template>
              </div>
            </at-collapse-item>
            <at-collapse-item class="mt-3" :title="checkpoint.name">
              <div class="checkpoint">
                <t-field class="checkpoint__item" inline label="Функция">
                  <t-select-new
                    :list="func"
                    small
                    update
                    name="metric_name"
                    :parse="'architecture[parameters][checkpoint][metric_name]'"
                    :value="getValue"
                    :disabled="disabled"
                    @parse="parse"
                  />
                </t-field>
                <template v-for="(data, i) of checkpoint.fields">
                  <t-auto-field-trainings
                    v-bind="data"
                    :key="'outputs_' + i"
                    class="checkpoint__item"
                    :state="state"
                    :inline="true"
                    :disabled="disabled"
                    @parse="parse"
                  />
                </template>
              </div>
            </at-collapse-item>
          </at-collapse>
        </div>
      </scrollbar>
    </div>
    <div class="params__footer">
      <div
        v-for="({ title, visible }, key) of button"
        :key="key"
        class="params__btn"
        :class="{ params__save: key === 'clear' }"
      >
        <t-button v-if="key !== 'save'" :disabled="!visible" @click="btnEvent(key)">{{ title }}</t-button>
      </div>
    </div>
  </div>
</template>

<script>
import { debounce } from '@/utils/core/utils';
import ser from '../../assets/js/myserialize';
import { mapGetters } from 'vuex';
import LoadSpiner from '@/components/forms/LoadSpiner';
export default {
  name: 'params-traning',
  components: {
    LoadSpiner,
  },
  data: () => ({
    collapse: [0, 1, 2, 3, 4],
    optimizerValue: '',
    metricData: '',
    debounce: null,
  }),
  computed: {
    ...mapGetters({
      params: 'trainings/getParams',
      button: 'trainings/getButtons',
      status: 'trainings/getStatus',
    }),
    isLearning() {
      return ['addtrain', 'training'].includes(this.status);
    },
    disabled() {
      return this.status !== 'no_train';
    },
    disabledAny() {
      const status = this.status;
      if (this.isLearning) {
        if (status === 'stopped') {
          return ['epochs'];
        }
        return true;
      }
      return false;
    },
    getValue() {
      let data = this.trainSettings?.architecture?.parameters?.outputs || [];
      data = data?.[this.metricData]?.metrics || [];
      this.saveValue(data);
      return this.state?.['architecture[parameters][checkpoint][metric_name]'] || data[0] || '';
    },
    state: {
      set(value) {
        this.$store.dispatch('trainings/setStateParams', value);
      },
      get() {
        return this.$store.getters['trainings/getStateParams'];
      },
    },
    trainSettings: {
      set(value) {
        this.$store.dispatch('trainings/setTrainSettings', value);
      },
      get() {
        return this.$store.getters['trainings/getTrainSettings'];
      },
    },
    main() {
      return this.params?.main || {};
    },
    fit() {
      return this.params?.fit || {};
    },
    outputs() {
      return this.params?.outputs || {};
    },
    optimizerFields() {
      return this.params?.optimizer?.fields?.[this.optimizerValue] || [];
    },
    optimizer() {
      return this.params?.optimizer || {};
    },
    checkpoint() {
      return this.params?.checkpoint || {};
    },
    statusTrain() {
      return this.$store.getters['trainings/getStatusTrain'];
    },
    func() {
      let data = this.trainSettings?.architecture?.parameters?.outputs || [];
      data = data?.[this.metricData]?.metrics || [];
      data = data.map(item => {
        return { label: item, value: item };
      });
      return data;
    },
  },
  methods: {
    saveValue([value]) {
      ser(this.trainSettings, 'architecture[parameters][checkpoint][metric_name]', value);
      this.trainSettings = { ...this.trainSettings };
    },
    btnEvent(key) {
      if (key === 'train') {
        this.start();
      }
      if (key === 'stop') {
        this.stop();
      }
      if (key === 'clear') {
        this.clear();
      }
      if (key === 'save') {
        this.save();
      }
    },
    click(e) {
      console.log(e);
    },
    async start() {
      // console.log(JSON.stringify(this.trainSettings, null, 2));
      const res = await this.$store.dispatch('trainings/start', this.trainSettings);
      if (res) {
        const { data } = res;
        if (data) {
          if (data?.state?.status) {
            localStorage.setItem('settingsTrainings', JSON.stringify(this.state));
            this.debounce(true);
          }
        }
      }
      // console.log(res);
    },
    async stop() {
      this.debounce(false);
      await this.$store.dispatch('trainings/stop', {});
    },
    async clear() {
      await this.$store.dispatch('trainings/clear', {});
    },
    async save() {
      await this.$store.dispatch('trainings/save', {});
    },
    async progress() {
      const res = await this.$store.dispatch('trainings/progress', {});
      if (res) {
        const { finished, message, percent } = res.data;
        this.$store.dispatch('messages/setProgressMessage', message);
        this.$store.dispatch('messages/setProgress', percent);
        if (!finished) {
          this.debounce(this.isLearning);
        } else {
          this.$store.dispatch('projects/get');
        }
      }
    },
    parse({ parse, value, name }) {
      // console.log({ parse, value, name });
      ser(this.trainSettings, parse, value);
      this.trainSettings = { ...this.trainSettings };
      if (name === 'architecture_parameters_checkpoint_layer') {
        this.metricData = value;
        if (value) {
          this.state = { [`${parse}`]: value };
        }
      } else {
        this.state = { [`${parse}`]: value };
      }
      if (name === 'optimizer') {
        this.optimizerValue = value;
      }
    },
  },
  created() {
    this.debounce = debounce(status => {
      // console.log(status);
      if (status) {
        this.progress();
      }
    }, 1000);

    // console.log(this.isLearning);
    this.debounce(this.isLearning);
    const settings = localStorage.getItem('settingsTrainings');
    if (settings) {
      try {
        this.state = JSON.parse(settings);
      } catch (error) {
        console.warn(error);
      }
    }
  },
  beforeDestroy() {
    this.debounce(false);
  },
};
</script>

<style lang="scss" scoped>
.blocks-layers {
  display: flex;
  flex-wrap: wrap;
}

.block-layers {
  width: 50%;
  &__header {
    color: #a7bed3;
    display: block;
    margin: 0 0 10px 0;
    line-height: 1;
    font-size: 0.75rem;
  }
}

.params {
  height: 100%;
  flex: 0 0 400px;
  border-left: #0e1621 solid 1px;
  display: flex;
  flex-direction: column;
  justify-content: space-between;
  position: relative;
  &__body {
    overflow: hidden;
    flex: 0 1 auto;
  }
  &__overlay {
    position: absolute;
    display: flex;
    align-items: center;
    justify-content: center;
    width: 100%;
    height: 100%;
    background-color: rgb(14 22 33 / 30%);
    z-index: 5;
  }
  &__footer {
    // width: 100%;
    padding: 10px 20px;
    display: flex;
    flex-wrap: wrap;
    // flex-direction: column;
    gap: 5%;
  }
  &__btn {
    width: 45%;
    margin: 0 0 10px 0;
  }
  &__save {
    width: 95%;
    margin: 0 0 10px 0;
  }
  &__items {
    padding-bottom: 20px;
    &--item {
      padding: 20px;
    }
  }
}

.fit,
.optimizer,
.checkpoint {
  display: flex;
  flex-wrap: wrap;
  &__item {
    width: 50%;
  }
}
</style>
