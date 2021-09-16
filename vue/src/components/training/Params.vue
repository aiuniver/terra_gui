<template>
  <div class="params">
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
                    :key="'optimizer_' + i"
                    class="optimizer__item"
                    :state="state"
                    inline
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
                          :key="'checkpoint_' + i"
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
            <at-collapse-item class="mt-3" :title="checkpoint.name">
              <div class="checkpoint">
                <t-field class="checkpoint__item" inline label="Функция">
                  <t-select-new :list="func" small name="metric_name" :parse="'architecture[parameters][checkpoint][metric_name]'" @parse="parse" />
                </t-field>
                <template v-for="(data, i) of checkpoint.fields">
                  <t-auto-field-trainings
                    v-bind="data"
                    :key="'outputs_' + i"
                    class="checkpoint__item"
                    :state="state"
                    :inline="true"
                    @parse="parse"
                  />
                </template>
              </div>
            </at-collapse-item>
          </at-collapse>
        </div>
        <!-- <div class="params__items">
        <div class="params__items--item">
          <t-field label="Мониторинг" inline>
              <TCheckbox small @focus="click" />
          </t-field>
        </div>
      </div> -->
      </scrollbar>
    </div>

    <div class="params__footer">
      <div>
        <t-button @click="start">Обучить</t-button>
        <t-button @click="stop">Остановить</t-button>
      </div>
      <div>
        <t-button @click="save">Сохранить</t-button>
        <t-button @click="clear">Сбросить</t-button>
      </div>
    </div>
  </div>
</template>

<script>
import ser from '../../assets/js/myserialize';
import { mapGetters } from 'vuex';
// import TCheckbox from '../global/new/forms/TCheckbox.vue';
// import Checkbox from '@/components/forms/Checkbox.vue';

export default {
  name: 'params-traning',
  components: {
    // TCheckbox,
    // Checkbox,
  },
  data: () => ({
    obj: {},
    collapse: [0, 1, 2, 3, 4],
    optimizerValue: '',
    metricData: ''
  }),
  computed: {
    ...mapGetters({
      params: 'trainings/getParams',
    }),
    state: {
      set(value) {
        this.$store.dispatch('trainings/setStateParams', value);
      },
      get() {
        // console.log(this.$store.getters['trainings/getStateParams']);
        return this.$store.getters['trainings/getStateParams'];
      },
    },
    main() {
      return this.params?.main || {};
    },
    fit() {
      return this.params?.fit || {};
    },
    outputs() {
      console.log(this.params?.outputs || {});
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
    func() {
      let data = this.obj?.architecture?.parameters?.outputs || [];
      data = data?.[this.metricData]?.metrics || [];
      data = data.map(item => {
        return { label: item, value: item };
      });
      return data;
    },
  },
  methods: {
    click(e) {
      console.log(e);
    },
    async start() {
      console.log(JSON.stringify(this.obj, null, 2));
      const res = await this.$store.dispatch('trainings/start', this.obj);
      console.log(res);
    },
    async stop() {
      const res = await this.$store.dispatch('trainings/stop', {});
      console.log(res);
    },
    async clear() {
      const res = await this.$store.dispatch('trainings/clear', {});
      console.log(res);
    },
    async save() {
      const res = await this.$store.dispatch('trainings/save', {});
      console.log(res);
    },
    parse({ parse, value, name }) {
      // console.log({ parse, value, name });
      this.state = { [`${parse}`]: value };
      ser(this.obj, parse, value);
      this.obj = { ...this.obj };
      if (name === 'architecture_parameters_checkpoint_layer') {
        this.metricData = value
      }
      if (name === 'optimizer') {
        this.optimizerValue = value;
      }
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
  &__body {
    overflow: hidden;
    flex: 0 1 auto;
  }
  &__footer {
    width: 100%;
    padding: 10px 20px;
    display: flex;
    flex-direction: column;
    gap: 10px;
    div {
      width: 100%;
      display: flex;
      gap: 10px;
    }
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