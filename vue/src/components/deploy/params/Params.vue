<template>
  <div class="params">
    <div v-if="false" class="params__overlay" key="fdgtr">
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
                  <t-auto-field-deploy
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
                        <t-auto-field-deploy
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
    <!-- <div class="params__footer">
      <div v-if="stopLearning" class="params__overlay">
        <LoadSpiner :text="'Остановка...'" />
      </div>
      <div v-for="({ title, visible }, key) of button" :key="key" class="params__btn">
        <t-button :disabled="!visible" @click="btnEvent(key)">{{ title }}</t-button>
      </div>
    </div> -->
  </div>

  <!-- <div class="params-container__name">Загрузка в демо-панель</div>
      <div class="params-container pa-5">
        <div class="t-input">
          <label class="label" for="deploy[deploy]">Название папки</label>
          <div class="t-input__label">
            https://srv1.demo.neural-university.ru/{{ userData.login }}/{{ projectData.name_alias }}/{{ deploy }}
          </div>
          <input
            v-model="deploy"
            class="t-input__input"
            type="text"
            id="deploy[deploy]"
            name="deploy[deploy]"
            @blur="$emit('blur', $event.target.value)"
          />
        </div>
        <Checkbox
          :label="'Перезаписать с таким же названием папки'"
          :type="'checkbox'"
          parse="deploy[overwrite]"
          name="deploy[overwrite]"
          class="pd__top"
          @change="UseReplace"
        />
        <Checkbox
          :label="'Использовать пароль для просмотра страницы'"
          parse="deploy[use_password]"
          name="deploy[use_password]"
          :type="'checkbox'"
          @change="UseSec"
        />
        <div class="password" v-if="use_sec">
          <div class="t-input">
            <input :type="passwordShow ? 'text' : 'password'" placeholder="Введите пароль" v-model="sec" />
            <div class="password__icon">
              <i
                :class="['t-icon', passwordShow ? 'icon-deploy-password-open' : 'icon-deploy-password-close']"
                :title="'show password'"
                @click="passwordShow = !passwordShow"
              ></i>
            </div>
          </div>
          <div class="t-input">
            <input :type="passwordShow ? 'text' : 'password'" placeholder="Подтверждение пароля" v-model="sec_accept" />
            <div class="password__icon">
              <i :class="['t-icon', checkCorrect]" :title="'is correct'"></i>
            </div>
          </div>
          <div class="password__rule">
            <p>Пароль должен содержать не менее 6 символов</p>
          </div>
        </div>
        <button :disabled="send_disabled" @click="SendData" v-if="!DataSent">Загрузить</button>
        <div class="loader" v-if="DataLoading">
          <div class="loader__title">Дождитесь окончания загрузки</div>
          <div class="loader__progress">
            <load-spiner></load-spiner>
          </div>
        </div>
        <div class="req-ans" v-if="DataSent">
          <div class="answer__success">Загрузка завершена!</div>
          <div class="answer__label">Ссылка на сформированную загрузку</div>
          <div class="answer__url">
            <i :class="['t-icon', 'icon-deploy-copy']" :title="'copy'" @click="Copy(moduleList.url)"></i>
            <a :href="moduleList.url" target="_blank">{{ moduleList.url }}sdfasadfasdfasgdfhasiofhusduifhasiodcfuisfhoadsifisdhfiosdup</a>
          </div>
        </div>
        <ModuleList v-if="DataSent" :moduleList="moduleList.api_text" />
      </div> -->
</template>

<script>
import { mapGetters } from 'vuex';
// import Checkbox from '@/components/forms/Checkbox';
// import ModuleList from './ModuleList';
// import LoadSpiner from '../../forms/LoadSpiner';
// import ser from '@/assets/js/myserialize';
export default {
  name: 'Settings',
  components: {
    // Checkbox,
    // ModuleList,
    // LoadSpiner,
  },
  data: () => ({
    collapse: ['type', 'server'],
    key: '1212',
    trainSettings: {},
    deploy: '',
    replace: false,
    use_sec: false,
    sec: '',
    sec_accept: '',
    DataSent: false,
    DataLoading: false,
    passwordShow: false,
    ops: {
      scrollPanel: {
        scrollingX: false,
        scrollingY: true,
      },
    },
  }),
  computed: {
    ...mapGetters({
      params: 'deploy/getParams',
      height: 'settings/height',
      moduleList: 'deploy/getModuleList',
      projectData: 'projects/getProject',
      userData: 'projects/getUser',
    }),
    state: {
      set(value) {
        this.$store.dispatch('deploy/setStateParams', value);
      },
      get() {
        return this.$store.getters['deploy/getStateParams'];
      },
    },
    checkCorrect() {
      return this.sec == this.sec_accept ? 'icon-deploy-password-correct' : 'icon-deploy-password-incorrect';
    },
    send_disabled() {
      if (this.DataLoading) {
        return true;
      }
      if (this.use_sec) {
        if (this.sec == this.sec_accept && this.sec.length > 5 && this.deploy.length != 0) return false;
      } else {
        if (this.deploy.length != 0) return false;
      }
      return true;
    },
  },
  methods: {
    parse(parse, value, changeable, mounted) {
      console.log(parse);
      console.log(parse, value, changeable, mounted);
      // ser(this.trainSettings, parse, value);
      // this.trainSettings = { ...this.trainSettings };
      if (!mounted && changeable) {
        // this.$store.dispatch('trainings/update', this.trainSettings);
        // this.state = { [`architecture[parameters][checkpoint][metric_name]`]: null };
      } else {
        if (value) {
          this.state = { [`${parse}`]: value };
        }
      }
    },
    onchange(e) {
      console.log(e);
      // console.log(this.collapse);
    },
    click() {
      console.log();
    },
    Percents(number) {
      let loading = document.querySelector('.progress-bar > .loading');
      loading.style.width = number + '%';
      loading.find('span').value = number;
    },
    Copy(text) {
      var textArea = document.createElement('textarea');
      textArea.value = text;

      textArea.style.top = '0';
      textArea.style.left = '0';
      textArea.style.position = 'fixed';

      document.body.appendChild(textArea);
      textArea.focus();
      textArea.select();

      try {
        document.execCommand('copy');
      } catch (err) {
        console.error('Fallback: Oops, unable to copy', err);
      }

      document.body.removeChild(textArea);
    },
    UseSec(data) {
      this.use_sec = data.value;
    },
    UseReplace(data) {
      this.replace = data.value;
    },
    async progress() {
      let answer = await this.$store.dispatch('deploy/CheckProgress');
      console.log(answer);
      if (!answer) {
        // this.Percents(30);
        this.getProgress();
      } else {
        this.DataLoading = false;
        this.DataSent = true;
        this.$emit('overlay', this.DataLoading);
      }
    },
    getProgress() {
      setTimeout(this.progress, 2000);
    },
    async SendData() {
      let data = {
        deploy: this.deploy,
        replace: this.replace,
        use_sec: this.use_sec,
      };

      if (this.use_sec) data['sec'] = this.sec;

      const res = await this.$store.dispatch('deploy/SendDeploy', data);
      console.log(res);
      if (res) {
        const { error, success } = res;
        console.log(error, success);
        if (!error && success) {
          this.DataLoading = true;
          this.$emit('overlay', this.DataLoading);
          this.getProgress();
        }
      }
    },
  },
  watch: {
    params() {
      this.key = 'dsdsdsd';
    },
  },
};
</script>

<style lang="scss" scoped>
.params {
  flex: 0 0 400px;
  border-left: #0e1621 solid 1px;
}
.params-container__name {
  padding: 30px 0 0 20px;
}
.pd__top {
  padding: 10px 0;
}
.label {
  color: #a7bed3;
  font-size: 12px;
  line-height: 24px;
}
.color__grey {
  color: #6c7883;
}
button {
  font-size: 0.875rem;
  margin-top: 10px;
  width: 107px;
}
.t-input {
  padding-bottom: 10px;
}
.loader {
  padding-top: 20px;
  &__title {
    font-size: 14px;
    line-height: 24px;
    text-align: center;
  }
  &__time {
    padding-top: 20px;
    font-size: 12px;
    line-height: 24px;
    color: #a7bed3;
    width: 100%;
    text-align: center;
  }
  &__progress {
    padding-top: 20px;
    width: 100%;
    display: flex;
    justify-content: center;
  }
}

.progress-bar {
  width: 426px;
  background: #2b5278;
  border-radius: 4px;
}
.loading {
  background: #65b9f4;
  border-radius: 4px;
  width: 78%;
  span {
    position: relative;
    margin-left: 201px;
  }
}
.t-input {
  &__label {
    color: #6c7883;
    display: block;
    margin: 0 0 10px 0;
    line-height: 1.25;
    font-size: 0.75rem;
    user-select: none;
    white-space: nowrap;
    text-overflow: ellipsis;
    width: 350px;
    display: inline-block;
    vertical-align: middle;
    overflow: hidden;
  }
  &__input {
    color: #fff;
    border-color: #6c7883;
    background: #242f3d;
    width: 100%;
  }
}
.req-ans {
  .answer__success {
    font-size: 14px;
    line-height: 24px;
    color: #ffffff;
    padding-top: 10px;
  }
  .answer__label {
    color: #a7bed3;
    font-size: 12px;
    line-height: 24px;
    padding-top: 15px;
  }
  .answer__url {
    font-size: 14px;
    line-height: 24px;
    color: #65b9f4;
    display: flex;
    a {
      padding-left: 10px;
      color: #65b9f4;
      white-space: nowrap;
      text-overflow: ellipsis;
      width: 350px;
      display: inline-block;
      vertical-align: middle;
      overflow: hidden;
    }
    i {
      cursor: pointer;
      width: 32px;
    }
  }
}
.password {
  &__icon {
    position: absolute;
    width: 345px;
    i {
      position: relative;
      float: right;
      margin-top: -34px;
    }
  }
  &__rule {
    p {
      font-size: 12px;
    }
  }
}
</style>
