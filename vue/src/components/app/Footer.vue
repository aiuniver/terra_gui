<template>
  <div class="footer">
    <div class="footer__message">
      <div class="footer__message--icon" @click="dialogErrors = true">
        <span></span>
      </div>
      <div :class="['footer__message--text', showMsg.color]" @click="dialogError = true">
        <transition name="error-slide" mode="out-in">
          <span :key="key">{{ showMsg.msg }}</span>
        </transition>
      </div>
    </div>
    <div class="footer__progress">
      <div class="footer__progress--item">
        <i :style="{ width: progress + '%' }">
          <span>
            {{ progressMessage }}
          </span>
        </i>
        <span>{{ progressMessage }}</span>
      </div>
    </div>
    <div class="footer__state">
      <div class="footer__state--text">
        <span :style="style"></span>
        {{ protsessor.type }}
      </div>
    </div>
    <div class="footer__copyright">
      {{ `Copyright © «Университет искусственного интеллекта», ${new Date().getFullYear()}` }}
      <span v-if="version" class="footer__version">{{ version }}</span>
    </div>
    <LoggingModal v-model="dialogErrors" :errors="errors" :title="'Логи'" @error="clickError" />
    <ErrorModal v-model="dialogLog" v-bind="logError" />
    <ErrorModal v-model="dialogError" v-bind="error" />
  </div>
</template>

<script>
import { mapGetters } from 'vuex';
import ErrorModal from '../global/modals/ErrorModal';
import LoggingModal from '../global/modals/LoggingModal';
export default {
  components: {
    ErrorModal,
    LoggingModal,
  },
  data: () => ({
    dialogError: false,
    dialogErrors: false,
    dialogLog: false,
    logError: {},
    text: '',
    key: 0,
    msgList: [],
    type: {
      info: 'Успешно',
      warning: 'Внимание',
      error: 'Ошибка',
      critical: 'Опасно',
    },
  }),
  computed: {
    ...mapGetters({
      message: 'messages/getMessage',
      color: 'messages/getColor',
      progress: 'messages/getProgress',
      project: 'projects/getProject',
      progressMessage: 'messages/getProgressMessage',
      error: 'logging/getError',
      logs: 'logging/getLogs',
      errors: 'logging/getErrors',
    }),
    protsessor() {
      return this.project?.hardware || '';
    },
    style() {
      return { backgroundColor: '#' + this.protsessor.color };
    },
    version() {
      return this.$config.isDev ? `ver. ${this.$config.version}` : '';
    },
    showMsg() {
      if (!this.msgList.length) return '';
      return this.msgList[0];
    },
  },
  methods: {
    click(color) {
      if (color === 'error') {
        this.text = this.msgList[0].msg;
        this.dialogError = true;
      }
    },
    clickError(error) {
      this.logError = error;
      this.dialogLog = true;
    },
  },
  watch: {
    error(value) {
      if (value) this.dialogError = true;
    },
    logs(arr) {
      if (arr.length) {
        arr.forEach(({ type = '', title = '', message = '', duration = 5000 }) => {
          if (type && title) {
            setTimeout(() => {
              this.$Notify({ type, title, message, duration });
            }, 100);
          }
        });
      }
    },
    // message(value) {
    //   if (value)
    //     // this.$Notify({ type: this.color, title: this.color === 'success' ? 'Успешно' : 'Ошибка', message: value });
    // },
  },
};
</script>

<style lang="scss" scoped>
.error {
  color: #ffb054 !important;
  cursor: pointer;
}
.info {
  color: #a7bed3 !important;
  cursor: pointer;
}
.success {
  color: #0f0 !important;
}
.footer {
  border-top: solid 1px #0e1621;
}
.state {
  text-transform: uppercase;
}

.footer {
  color: #a7bed3;
  background: #17212b;
  width: 100%;
  line-height: 30px;
  position: fixed;
  left: 0;
  bottom: 30px;
  z-index: 900;
  font-size: 0.75rem;
  white-space: nowrap;
  user-select: none;
  display: flex;
  flex-direction: row;
  flex-wrap: nowrap;
  justify-content: flex-start;
  align-content: stretch;
  align-items: stretch;
  &__version {
    color: ivory;
  }
  &__copy-buffer {
    display: flex;
    align-items: center;
    p {
      margin-left: 5px;
    }
  }
  &__message {
    border-right: #0e1621 1px solid;
    overflow: hidden;
    order: 0;
    flex: 1 1 auto;
    align-self: auto;
    display: flex;
    &--text {
      overflow: hidden;
      text-overflow: ellipsis;
      padding: 0 10px;
      span {
        display: block;
        overflow: hidden;
        text-overflow: ellipsis;
        transition-duration: 200ms;
      }
    }
    &--icon {
      cursor: pointer;
      display: flex;
      justify-content: center;
      align-items: center;
      flex: 0 0 40px;
      span {
        display: flex;
        width: 10px;
        height: 10px;
        border-radius: 50%;
        background-color: #ffb054;
      }
    }
  }
  &__state {
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 600;
    &--text {
      text-transform: uppercase;
      padding: 0 10px;
      width: 59px;
      & > span {
        display: inline-block;
        content: '';
        width: 10px;
        height: 10px;
        margin: 0 5px 0 0;
        border-radius: 50%;
        vertical-align: middle;
      }
    }
  }
  &__progress {
    border-right: #0e1621 1px solid;
    &--item {
      width: 339px;
      height: 100%;
      padding: 3px;
      position: relative;
      > i {
        background-color: #242f3d;
        display: block;
        width: 0;
        height: 100%;
        position: relative;
        z-index: 1;
        font-style: normal;
        overflow: hidden;
        > span {
          color: #fff;
          line-height: 24px;
          position: absolute;
          left: 0;
          top: 0;
          padding: 0 7px;
        }
      }
      > span {
        color: #2b5278;
        display: block;
        line-height: 30px;
        padding: 0 10px;
        position: absolute;
        left: 0;
        top: 0;
        z-index: 0;
        width: 100%;
        white-space: nowrap;
      }
    }
  }
  &__copyright {
    position: fixed;
    width: 100%;
    left: 0;
    bottom: 0;
    z-index: 901;
    user-select: none;
    line-height: 30px;
    padding: 0 10px;
    color: #a7bed3;
    font-size: 0.6875rem;
    text-align: right;
    background-color: #0e1621;
  }
}

.error-slide-leave-active {
  transform: translateY(0);
}
.error-slide-leave-to {
  transform: translateY(-100%);
  opacity: 0;
}
.error-slide-enter {
  transform: translateY(100%);
}
</style>
