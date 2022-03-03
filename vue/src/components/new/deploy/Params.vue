<template>
  <div class="params deploy" :key="'key_update-' + updateKey">
    <scrollbar>
      <div class="params__body">
        <div class="params__items">
          <at-collapse :value="collapse">
            <at-collapse-item
              v-for="({ visible, name, fields }, key) of params"
              v-show="visible && key !== 'server'"
              :key="key"
              class="mt-3"
              :name="key"
              :title="name || ''"
            >
              <div class="params__fields">
                <template v-for="(data, i) of fields">
                  <TAutoFieldCascade
                    v-bind="data"
                    :key="key + i"
                    :big="key === 'type'"
                    :parameters="parameters"
                    :inline="false"
                    @change="parse"
                  />
                  <d-button @click="onStart" :key="'key' + i" :disabled="overlayStatus" style="margin-top: 20px;">Подготовить</d-button>
                </template>
              </div>
            </at-collapse-item>
          </at-collapse>
        </div>
        <div class="params__items" v-if="paramsDownloaded.isParamsSettingsLoad">
          <div class="params-container pa-5">
            <div class="t-input">
              <label class="label" for="deploy[deploy]">Название папки</label>
              <div class="t-input__label">
                {{ `https://srv1.demo.neural-university.ru/${userData.login}/${projectData.name_alias}/${deploy}` }}
              </div>
              <input v-model="deploy" class="t-input__input" type="text" id="deploy[deploy]" name="deploy[deploy]" autocomplete="off" />
            </div>
            <DAutocomplete
              autocomplete="off"
              :value="serverLabel"
              :list="list"
              :name="'deploy[server]'"
              label="Сервер"
              @focus="focus"
              @change="selected"
            />

            <Checkbox
              :label="'Перезаписать с таким же названием папки'"
              :type="'checkbox'"
              parse="replace"
              name="replace"
              class="pd__top"
              @change="onChange"
            />
            <Checkbox
              :label="'Использовать пароль для просмотра страницы'"
              parse="replace"
              name="use_sec"
              :type="'checkbox'"
              @change="onChange"
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
                <input
                  :type="passwordShow ? 'text' : 'password'"
                  placeholder="Подтверждение пароля"
                  v-model="sec_accept"
                />
                <div class="password__icon">
                  <i :class="['t-icon', checkCorrect]" :title="'is correct'"></i>
                </div>
              </div>
              <div class="password__rule">
                <p>Пароль должен содержать не менее 6 символов</p>
              </div>
            </div>
            <d-button style="margin-top: 20px;" :disabled="send_disabled" @click="sendDeployData" v-if="!paramsDownloaded.isSendParamsDeploy">
              Загрузить
            </d-button>
            <div class="req-ans" v-if="paramsDownloaded.isSendParamsDeploy">
              <div class="answer__success">Загрузка завершена!</div>
              <div class="answer__label">Ссылка на сформированную загрузку</div>
              <div class="answer__url">
                <i :class="['t-icon', 'icon-deploy-copy']" :title="'copy'" @click="copy(moduleList.url)"></i>
                <a :href="moduleList.url" target="_blank">
                  {{ moduleList.url }}
                </a>
              </div>
            </div>
            <ModuleList v-if="paramsDownloaded.isSendParamsDeploy" :moduleList="moduleList.api_text" />
          </div>
        </div>
      </div>
    </scrollbar>
  </div>
</template>

<script>
import Checkbox from '@/components/new/forms/Checkbox';
import DAutocomplete from '@/components/new/forms/DAutocomplete';
import ModuleList from '@/components/deploy/params/ModuleList';
import TAutoFieldCascade from '@/components/new/blocks/TAutoFieldCascade';
import { DEPLOY_ICONS_PASSWORD, DEPLOY_COLLAPS } from '@/components/deploy/config/const-params';

export default {
  name: 'Settings',
  components: {
    Checkbox,
    ModuleList,
    DAutocomplete,
    TAutoFieldCascade
  },
  props: {
    params: {
      type: [Object, Array],
      default: () => ({}),
    },
    moduleList: {
      type: Object,
      default: () => ({}),
    },
    projectData: {
      type: [Array, Object],
      default: () => [],
    },
    userData: {
      type: [Object, Array],
      default: () => {},
    },
    paramsDownloaded: {
      type: Object,
      default: () => ({}),
    },
    overlayStatus: {
      type: Boolean,
      default: false,
    },
  },
  data: () => ({
    updateKey: 0,
    collapse: DEPLOY_COLLAPS,
    deploy: '',
    server: '',
    serverLabel: '',
    replace: false,
    use_sec: false,
    sec: '',
    sec_accept: '',
    passwordShow: false,
    parameters: {},
    list: [],
  }),
  computed: {
    checkCorrect() {
      return this.sec == this.sec_accept ? DEPLOY_ICONS_PASSWORD[0] : DEPLOY_ICONS_PASSWORD[1];
    },
    send_disabled() {
      if (this.use_sec && this.sec == this.sec_accept && this.sec.length > 5 && this.deploy.length != 0) return false;
      else if (this.deploy.length != 0) return false;
      return true;
    },
    isLoad() {
      return !(!!this.parameters.type && !!this.parameters.name);
    },
  },
  methods: {
    parse({ value, name }) {
      if (name === 'type') this.parameters['name'] = null;
      this.parameters[name] = value;
      this.parameters = { ...this.parameters };
    },
    onChange({ name, value }) {
      this[name] = value;
    },
    copy(text) {
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
    async onStart() {
      this.$emit('downloadSettings', this.parameters);
      await this.focus();
    },
    async focus() {
      const res = await this.$store.dispatch('servers/ready');
      if (res.data) this.list = res?.data || [];
      const { value, label } = this.list?.[0];
      if (!this.serverLabel) {
        this.serverLabel = label;
        this.server = value;
      }
    },
    selected({ value }) {
      this.server = value;
    },
    async sendDeployData() {
      const data = {
        deploy: this.deploy,
        server: this.server,
        replace: this.replace,
        use_sec: this.use_sec,
      };
      if (this.use_sec) data['sec'] = this.sec;
      this.$emit('sendParamsDeploy', data);
    },
  },
  beforeDestroy() {
    this.$emit('clear');
  },
  watch: {
    params() {
      this.updateKey++;
    },
  },
};
</script>

<style lang="scss" scoped>
.params {
  flex: 0 0 400px;
  border-left: #0e1621 solid 1px;
  &__fields {
    button {
      margin: 30px 0 0 0;
    }
  }
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
    border-color: #65B9F4;
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

input {
  border-color: #65B9F4;
}

input:focus {
  border-color: #65B9F4;
  background: rgba(101, 185, 244, 0.15);
}
</style>
