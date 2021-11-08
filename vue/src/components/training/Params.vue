<template>
  <div class="params">
    <div v-if="statusTrain === 'start'" class="params__overlay" key="fdgtr">
      <LoadSpiner :text="'Запуск обучения...'" />
    </div>
    <scrollbar>
      <div class="params__body">
        <div class="params__items">
          <at-collapse :value="collapse" @on-change="onchange" :key="key">
            <at-collapse-item
              v-show="visible"
              v-for="({ visible, name, fields }, key) of params"
              :key="key"
              class="mt-3"
              :name="key"
              :title="name || ''"
            >
              <div v-if="key !== 'outputs'" class="params__fields">
                <template v-for="(data, i) of fields">
                  <t-auto-field-trainings
                    v-bind="data"
                    :class="`params__fields--${key}`"
                    :key="key + i"
                    :state="state"
                    :inline="false"
                    @parse="parse"
                  />
                </template>
              </div>
              <div v-else class="blocks-layers">
                <template v-for="(field, i) of fields">
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
                          @parse="parse"
                        />
                      </template>
                    </div>
                  </div>
                </template>
              </div>
            </at-collapse-item>
          </at-collapse>
        </div>
      </div>
    </scrollbar>
    <div class="params__footer">
      <div v-if="stopLearning" class="params__overlay">
        <LoadSpiner :text="'Остановка...'" />
      </div>
      <div
        v-for="({ title, visible }, key) of button"
        :key="key"
        class="params__btn"
      >
        <t-button :disabled="!visible" @click="btnEvent(key)">{{ title }}</t-button>
      </div>
      <!-- <div class="params__btn">
        <t-button :disabled="false" @click="btnEvent('save')">{{ 'Сохранить' }}</t-button>
      </div> -->
    </div>
    <SaveTrainings v-model="dialogSave" />
  </div>
</template>

<script>
import { debounce } from '@/utils/core/utils';
import ser from '../../assets/js/myserialize';
import { mapGetters } from 'vuex';
import LoadSpiner from '@/components/forms/LoadSpiner';
import SaveTrainings from '@/components/app/modal/SaveTrainings';
export default {
  name: 'params-traning',
  components: {
    LoadSpiner,
    SaveTrainings,
  },
  data: () => ({
    collapse: ['main', 'fit', 'outputs', 'checkpoint', 'yolo'],
    optimizerValue: '',
    metricData: '',
    debounce: null,
    stopLearning: false,
    dialogSave: false,
    trainSettings: {},
    key: '1212',
    doNotSave: ['architecture[parameters][checkpoint][metric_name]'],
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
    statusTrain() {
      return this.$store.getters['trainings/getStatusTrain'];
    },
    state: {
      set(value) {
        this.$store.dispatch('trainings/setStateParams', value);
      },
      get() {
        return this.$store.getters['trainings/getStateParams'];
      },
    },
  },
  methods: {
    onchange(e) {
      console.log(e);
      // console.log(this.collapse);
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
    async start() {
      console.log(this.trainSettings)
      const res = await this.$store.dispatch('trainings/start', this.trainSettings);
      if (res) {
        const { data } = res;
        if (data) {
          if (data?.state?.status) {
            this.debounce(true);
          }
        }
      }
    },
    async stop() {
      this.stopLearning = true;
      const res = await this.$store.dispatch('trainings/stop', {});
      if (res && res?.data?.progress) {
        const { finished } = res.data.progress;
        if (finished) {
          this.debounce(false);
          this.stopLearning = false;
        }
      }
    },
    async clear() {
      await this.$store.dispatch('trainings/clear', {});
    },
    save() {
      this.dialogSave = true;
      // await this.$store.dispatch('trainings/save', {});
    },
    async progress() {
      const res = await this.$store.dispatch('trainings/progress', {});
      // console.log(res?.data?.progress)
      if (res && res?.data?.progress) {
        const { finished, message, percent } = res.data.progress;
        this.$store.dispatch('messages/setProgressMessage', message);
        this.$store.dispatch('messages/setProgress', percent);
        this.stopLearning = !this.isLearning;
        if (!finished) {
          this.debounce(true);
        } else {
          this.$store.dispatch('projects/get');
          this.stopLearning = false;
        }
      }
      if (res?.error) {
        this.stopLearning = false;
      }
    },
    parse({ parse, value, changeable, mounted }) {
      // parse({ parse, value, name, changeable, mounted }) {
      // console.log({ parse, value, name, changeable, mounted });
      ser(this.trainSettings, parse, value);
      this.trainSettings = { ...this.trainSettings };
      if (!mounted && changeable) {
        this.$store.dispatch('trainings/update', this.trainSettings);
        this.state = { [`architecture[parameters][checkpoint][metric_name]`]: null };
      } else {
        if (value) {
          this.state = { [`${parse}`]: value };
        }
      }
    },
  },
  created() {
    this.debounce = debounce(status => {
      if (status) {
        this.progress();
      }
    }, 1000);
    this.debounce(this.isLearning);
  },
  beforeDestroy() {
    this.debounce(false);
  },
  watch: {
    params() {
      this.key = 'dsdsdsd';
    },
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
    position: relative;
    // width: 100%;
    margin: 0 20px;
    padding: 10px 0;
    display: flex;
    flex-wrap: wrap;
    // flex-direction: column;
    gap: 2%;
  }
  &__btn {
    width: 49%;
    margin: 0 0 10px 0;
  }
  &__save {
    width: 100%;
    margin: 0 0 10px 0;
  }
  &__items {
    padding-bottom: 20px;
  }
  &__fields {
    display: flex;
    flex-wrap: wrap;
    div {
      width: 50%;
    }
    &--main {
      width: 100% !important;
    }
  }
}

.btn-spiner {
  margin-top: 10px;
}
</style>
