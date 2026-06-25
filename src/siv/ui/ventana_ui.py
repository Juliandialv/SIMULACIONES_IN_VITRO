# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'ventana_ui.ui'
##
## Created by: Qt User Interface Compiler version 6.11.0
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QBrush, QColor, QConicalGradient, QCursor,
    QFont, QFontDatabase, QGradient, QIcon,
    QImage, QKeySequence, QLinearGradient, QPainter,
    QPalette, QPixmap, QRadialGradient, QTransform)
from PySide6.QtWidgets import (QApplication, QHBoxLayout, QLabel, QMainWindow,
    QMenuBar, QPushButton, QSizePolicy, QSpacerItem,
    QTextEdit, QVBoxLayout, QWidget)

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(1440, 880)
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.verticalLayout_main = QVBoxLayout(self.centralwidget)
        self.verticalLayout_main.setSpacing(0)
        self.verticalLayout_main.setObjectName(u"verticalLayout_main")
        self.label_picking_hint = QLabel(self.centralwidget)
        self.label_picking_hint.setObjectName(u"label_picking_hint")
        sizePolicy = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_picking_hint.sizePolicy().hasHeightForWidth())
        self.label_picking_hint.setSizePolicy(sizePolicy)
        self.label_picking_hint.setMaximumSize(QSize(16777215, 36))
        self.label_picking_hint.setVisible(False)
        self.label_picking_hint.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.verticalLayout_main.addWidget(self.label_picking_hint)

        self.viewport_container = QWidget(self.centralwidget)
        self.viewport_container.setObjectName(u"viewport_container")
        sizePolicy1 = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        sizePolicy1.setHorizontalStretch(0)
        sizePolicy1.setVerticalStretch(10)
        sizePolicy1.setHeightForWidth(self.viewport_container.sizePolicy().hasHeightForWidth())
        self.viewport_container.setSizePolicy(sizePolicy1)

        self.verticalLayout_main.addWidget(self.viewport_container)

        self.action_bar = QWidget(self.centralwidget)
        self.action_bar.setObjectName(u"action_bar")
        sizePolicy.setHeightForWidth(self.action_bar.sizePolicy().hasHeightForWidth())
        self.action_bar.setSizePolicy(sizePolicy)
        self.action_bar.setMaximumSize(QSize(16777215, 70))
        self.action_bar.setVisible(False)
        self.horizontalLayout_actions = QHBoxLayout(self.action_bar)
        self.horizontalLayout_actions.setObjectName(u"horizontalLayout_actions")
        self.horizontalSpacer_left = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_actions.addItem(self.horizontalSpacer_left)

        self.qpushButton_landmarks = QPushButton(self.action_bar)
        self.qpushButton_landmarks.setObjectName(u"qpushButton_landmarks")
        self.qpushButton_landmarks.setMinimumSize(QSize(200, 48))
        self.qpushButton_landmarks.setMaximumSize(QSize(300, 48))

        self.horizontalLayout_actions.addWidget(self.qpushButton_landmarks)

        self.qpushButton_compute = QPushButton(self.action_bar)
        self.qpushButton_compute.setObjectName(u"qpushButton_compute")
        self.qpushButton_compute.setMinimumSize(QSize(200, 48))
        self.qpushButton_compute.setMaximumSize(QSize(300, 48))
        self.qpushButton_compute.setVisible(False)

        self.horizontalLayout_actions.addWidget(self.qpushButton_compute)

        self.qpushButton_uncertainty = QPushButton(self.action_bar)
        self.qpushButton_uncertainty.setObjectName(u"qpushButton_uncertainty")
        self.qpushButton_uncertainty.setMinimumSize(QSize(200, 48))
        self.qpushButton_uncertainty.setMaximumSize(QSize(300, 48))
        self.qpushButton_uncertainty.setVisible(False)

        self.horizontalLayout_actions.addWidget(self.qpushButton_uncertainty)

        self.qpushButton_automatic = QPushButton(self.action_bar)
        self.qpushButton_automatic.setObjectName(u"qpushButton_automatic")
        self.qpushButton_automatic.setMinimumSize(QSize(200, 48))
        self.qpushButton_automatic.setMaximumSize(QSize(300, 48))

        self.horizontalLayout_actions.addWidget(self.qpushButton_automatic)

        self.horizontalSpacer_right = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_actions.addItem(self.horizontalSpacer_right)


        self.verticalLayout_main.addWidget(self.action_bar)

        self.textEdit_log = QTextEdit(self.centralwidget)
        self.textEdit_log.setObjectName(u"textEdit_log")
        sizePolicy.setHeightForWidth(self.textEdit_log.sizePolicy().hasHeightForWidth())
        self.textEdit_log.setSizePolicy(sizePolicy)
        self.textEdit_log.setMaximumSize(QSize(16777215, 100))
        self.textEdit_log.setReadOnly(True)

        self.verticalLayout_main.addWidget(self.textEdit_log)

        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QMenuBar(MainWindow)
        self.menubar.setObjectName(u"menubar")
        self.menubar.setGeometry(QRect(0, 0, 1440, 17))
        MainWindow.setMenuBar(self.menubar)

        self.retranslateUi(MainWindow)

        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"SIV \u2014 Assimmetry Analysis", None))
        self.label_picking_hint.setText("")
        self.qpushButton_landmarks.setText(QCoreApplication.translate("MainWindow", u"  SELECT LANDMARKS", None))
        self.qpushButton_compute.setText(QCoreApplication.translate("MainWindow", u"  COMPUTE INDICES", None))
        self.qpushButton_uncertainty.setText(QCoreApplication.translate("MainWindow", u"  RUN UNCERTAINTY", None))
        self.qpushButton_automatic.setText(QCoreApplication.translate("MainWindow", u"  AUTOMATIC LANDMARKS", None))
    # retranslateUi

